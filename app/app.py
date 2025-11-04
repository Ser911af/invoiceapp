import io
import re
import numpy as np
import pandas as pd
import streamlit as st
from datetime import datetime
from supabase import create_client


# =========================
# === OPTIONAL: SUPABASE SDK PLACEHOLDER (lazy import in function)
# =========================
try:
    # We won't import here to avoid ModuleNotFoundError at import time.
    # The real import happens inside load_sales_with_diagnostics().
    from supabase import create_client  # type: ignore
except Exception:
    create_client = None  # We'll handle this in the loader function.


# =========================
# === UTILITIES
# =========================

def classify_invoice(invoice):
    s = "" if pd.isna(invoice) else str(invoice).strip()
    sl = s.lower()
    if "need" in sl:
        return "need"
    if "flag file" in sl or re.search(r"\bff\b", sl):
        return "flag_file"
    if re.search(r"\bok\b", sl):
        return "ok"
    if s == "" or re.fullmatch(r"[A-Za-z0-9_\-\/]+", s):
        return "empty_or_number"
    return "other"


def calculate_dias360(requested_date, ref_date):
    """Acepta datetime/fecha/string; devuelve días 360 o None si no parsea."""
    try:
        if pd.isna(requested_date):
            return None
        requested_date = pd.to_datetime(requested_date).date()
        diff = (
            (ref_date.year - requested_date.year) * 360
            + (ref_date.month - requested_date.month) * 30
            + (ref_date.day - requested_date.day)
        )
        return diff
    except Exception:
        return None


def coerce_money_series(s):
    if s is None:
        return s
    return (
        s.astype(str)
         .str.replace(r"[\$,]", "", regex=True)
         .str.replace(r"\s", "", regex=True)
         .replace({"": None})
         .pipe(pd.to_numeric, errors="coerce")
    )


def extract_digits(val):
    if pd.isna(val):
        return None
    m = re.search(r"(\d+)", str(val))
    return m.group(1) if m else None


def build_salesrep_lookup(consolidated_df: pd.DataFrame):
    dfc = consolidated_df.copy()
    dfc.columns = dfc.columns.str.strip()

    # Normalize to lower for flexible matching
    lower_map = {c.lower(): c for c in dfc.columns}
    source_col = lower_map.get("source") or lower_map.get("sales_order") or lower_map.get("Source")
    # Prefer 'sales_rep', fallback to 'cus_sales_rep'
    salesrep_col = lower_map.get("sales_rep") or lower_map.get("cus_sales_rep") or lower_map.get("Sales Rep")

    if not source_col or not salesrep_col:
        raise ValueError("Missing columns for mapping (expect 'source' and 'sales_rep' or 'cus_sales_rep').")

    dfc["po_key"] = dfc[source_col].apply(extract_digits)

    def first_non_empty(s):
        for x in s:
            if pd.notna(x) and str(x).strip() != "":
                return str(x).strip()
        return None

    salesrep_map = (
        dfc.sort_values(by=["po_key"])
           .groupby("po_key", dropna=True)[salesrep_col]
           .apply(first_non_empty)
           .to_dict()
    )
    return {k: v for k, v in salesrep_map.items() if k}


def _is_zero_amount(x):
    try:
        return np.isclose(float(x if pd.notna(x) else 0.0), 0.0)
    except Exception:
        return False


def _autofit_widths(worksheet, df_sheet, formats_by_col=None, min_w=9, max_w=42, pad=2, ratio=1.1):
    """Ajusta anchos de columnas al contenido aproximando número de caracteres."""
    for col_name in df_sheet.columns:
        col_idx = df_sheet.columns.get_loc(col_name)
        series = df_sheet[col_name].astype(str).replace("nan", "")
        max_len_cells = series.map(len).max() if len(series) else 0
        header_len = len(str(col_name))
        width = int(max(header_len, max_len_cells) * ratio) + pad
        width = max(min_w, min(width, max_w))
        fmt = formats_by_col.get(col_name) if formats_by_col else None
        worksheet.set_column(col_idx, col_idx, width, fmt)


# =========================
# === SUPABASE LOADER WITH DIAGNOSTICS (SDK + REST fallback)
# =========================

def load_sales_with_diagnostics() -> pd.DataFrame | None:
    """Loads the sales consolidated table from Supabase.
    Tries the official SDK first; falls back to REST if the SDK is not installed.
    Prints detailed diagnostics on failure.
    """
    cfg = st.secrets.get("supabase_sales", {}) if hasattr(st, "secrets") else {}
    url = cfg.get("url")
    key = cfg.get("key")
    table_name = cfg.get("table", "ventas_frutto")

    if not url or not key:
        st.error("❌ Supabase credentials missing in secrets.toml → [supabase_sales] (need url, key).")
        st.write(cfg)  # Shows what actually loaded from secrets
        return None

    # --- Try SDK path (lazy import inside) ---
    try:
        from supabase import create_client as _create_client  # type: ignore
        try:
            sb = _create_client(url, key)
            res = sb.table(table_name).select("source,sales_rep,cus_sales_rep").limit(100000).execute()
            rows = res.data or []
            if not rows:
                st.warning(f"⚠️ Connected via SDK but table '{table_name}' returned 0 rows.")
                return None
            st.success(f"✅ Supabase SDK connection successful. Rows: {len(rows):,}")
            return pd.DataFrame(rows)
        except Exception as e:
            st.exception(e)
            st.warning("Falling back to REST API due to SDK error…")
    except ModuleNotFoundError:
        st.info("Supabase SDK not installed. Falling back to REST API. Add 'supabase' to requirements.txt to use SDK.")

    # --- REST fallback ---
    try:
        import requests
        from urllib.parse import urljoin

        rest_base = urljoin(url if url.endswith("/") else url + "/", "rest/v1/")
        endpoint = urljoin(rest_base, f"{table_name}")
        params = {"select": "source,sales_rep,cus_sales_rep"}
        headers = {
            "apikey": key,
            "Authorization": f"Bearer {key}",
            "Accept": "application/json",
            "Range": "0-99999",
        }
        r = requests.get(endpoint, headers=headers, params=params, timeout=20)
        if r.status_code >= 400:
            st.error(f"❌ REST query failed ({r.status_code}): {r.text[:300]}")
            return None

        data = r.json()
        if not data:
            st.warning(f"⚠️ REST call succeeded but returned 0 rows from '{table_name}'.")
            return None

        st.success(f"✅ Supabase REST connection successful. Rows: {len(data):,}")
        return pd.DataFrame(data)
    except Exception as e:
        st.exception(e)
        st.error("❌ Could not fetch data via Supabase REST API.")
        return None


# =========================
# === STREAMLIT APP
# =========================
st.set_page_config(page_title="FL Rev Confirmed Invoices", page_icon="📄")

st.title("📄 FL Rev Confirmed Invoices – Processor")
st.markdown((
    "Upload the **Expenses CSV** (AP > Expenses). "
    "The app will automatically fetch the **sales consolidated** from **Supabase** "
    "(table `ventas_frutto`) to map the *Sales Rep*. "
    "If the connection or secret is missing, it will proceed without that column."
))

csv_file = st.file_uploader("Expenses CSV (ap-expenses-*.csv)", type=["csv"], accept_multiple_files=False)
debug_supabase = st.checkbox("Show Supabase diagnostics", value=False)
preview = st.checkbox("Show data preview", value=False)

if st.button("Process", type="primary"):
    if csv_file is None:
        st.error("You must upload the Expenses CSV.")
        st.stop()

    # =========================
    # === LOAD & PREPROCESS
    # =========================
    today = datetime.today().date()
    dias_col_name = datetime.today().strftime("%m/%d/%y")

    try:
        df = pd.read_csv(csv_file)
    except Exception as e:
        st.exception(e)
        st.stop()

    df.columns = df.columns.str.strip()

    required_cols = [
        "Requested Date", "Received Date", "Due Date", "PO # / EXP #",
        "Invoice Number", "Vendor Name", "Description", "Total Amount",
        "Buyer", "Status"
    ]
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        st.error(f"Missing required columns in CSV: {missing}")
        st.stop()

    # Numeric cleanup
    df["Total Amount"] = coerce_money_series(df["Total Amount"])

    # Parse dates to datetime (not strings yet)
    for date_col in ["Requested Date", "Received Date", "Due Date"]:
        df[date_col] = pd.to_datetime(df[date_col], errors="coerce")

    # Functional filter: ONLY Produce + (paid with 0) OR (unpaid/partially paid)
    desc_norm = df["Description"].astype(str).str.strip().str.lower()
    status_norm = df["Status"].astype(str).str.strip().str.lower()

    mask_produce = (desc_norm == "produce")
    mask_paid_zero = (status_norm == "paid") & (df["Total Amount"].fillna(0) == 0)
    mask_unpaid_or_partial = status_norm.isin(["unpaid", "partially paid"])

    df = df[mask_produce & (mask_paid_zero | mask_unpaid_or_partial)].copy()

    if df.empty:
        st.warning("Filtering returned 0 rows. Check the CSV (Description must be 'Produce').")
        st.stop()

    # Classification & metrics
    df["Invoice Group"] = df["Invoice Number"].apply(classify_invoice)
    df[dias_col_name] = df["Requested Date"].apply(lambda x: calculate_dias360(x, today))
    df["Answer"] = ""

    # Drop Status
    df.drop(columns=["Status"], inplace=True, errors="ignore")

    # Format dates (to strings mm/dd/yy)
    for date_col in ["Requested Date", "Received Date", "Due Date"]:
        df[date_col] = df[date_col].dt.strftime("%m/%d/%y")

    columns_needed = [
        "Requested Date", dias_col_name, "Received Date", "Due Date",
        "PO # / EXP #", "Invoice Number", "Vendor Name", "Description",
        "Total Amount", "Buyer", "Answer"
    ]
    df = df[[c for c in columns_needed if c in df.columns] + ["Invoice Group"]].copy()

    # =========================
    # === SALES REP MAPPING (Supabase)
    # =========================
    salesrep_map = None
    sales_df = load_sales_with_diagnostics()
    if sales_df is not None and len(sales_df) > 0:
        try:
            salesrep_map = build_salesrep_lookup(sales_df)
            if salesrep_map:
                st.success(f"Supabase connected • Rows: {len(sales_df):,} • Unique PO keys mapped: {len(salesrep_map):,}")
            else:
                st.warning("Supabase connected but no PO keys could be mapped. Ensure 'source' contains the PO/EXP number.")
            if debug_supabase:
                st.caption("Supabase sample (head):")
                st.dataframe(sales_df.head(50))
        except Exception as e:
            st.warning(f"Could not build Sales Rep map from Supabase: {e}")
            salesrep_map = None
    else:
        st.info("Supabase not configured or returned 0 rows. Proceeding without Sales Rep column.")

    # =========================
    # === GROUPING
    # =========================
    flag_mask_zero = df["Total Amount"].apply(_is_zero_amount)
    flag_mask_class = (df["Invoice Group"] == "flag_file")
    flag_mask_pas = df["Invoice Number"].astype(str).str.contains(r"PAS", case=False, na=False)

    # New logic: if it's "need" and zero amount, do NOT send to Flag_File
    need_mask = (df["Invoice Group"] == "need")
    flag_union_mask = (flag_mask_class | flag_mask_pas | (flag_mask_zero & ~need_mask))

    groups = {
        "Flag_File":       df[flag_union_mask].copy(),
        "Empty_or_Number": df[(df["Invoice Group"] == "empty_or_number") & (~flag_union_mask)].copy(),
        "Need":            df[(df["Invoice Group"] == "need") & (~flag_union_mask)].copy(),
    }

    # In Flag_File → Sales Rep
    if salesrep_map is not None and not groups["Flag_File"].empty:
        po_keys = groups["Flag_File"]["PO # / EXP #"].apply(extract_digits)
        groups["Flag_File"]["Sales Rep"] = po_keys.map(salesrep_map).fillna("")
        if "Answer" in groups["Flag_File"].columns:
            groups["Flag_File"].drop(columns=["Answer"], inplace=True)

    # Remove 'Answer' from non-Flag_File sheets
    for grp_name in ["Empty_or_Number", "Need"]:
        if not groups[grp_name].empty and "Answer" in groups[grp_name].columns:
            groups[grp_name].drop(columns=["Answer"], inplace=True)

    # =========================
    # === EXPORT TO EXCEL
    # =========================
    today_str_header = datetime.today().strftime("%B %d").upper()
    output_filename = f"FL REV CONFIRMED INVOICES {today_str_header}.xlsx"

    xlsx_buffer = io.BytesIO()
    with pd.ExcelWriter(xlsx_buffer, engine="xlsxwriter") as writer:
        for name, sheet in groups.items():
            sheet = sheet.drop(columns=["Invoice Group"], errors="ignore")
            sheet.to_excel(writer, index=False, sheet_name=name)

            workbook = writer.book
            worksheet = writer.sheets[name]

            date_format = workbook.add_format({"num_format": "mm/dd/yyyy"})
            money_format_base = {"num_format": "$#,##0.00"}
            yellow_fill = workbook.add_format({"bg_color": "#FFFF00"})
            red_fill_white = workbook.add_format({"bg_color": "#FF0000", "font_color": "#FFFFFF"})
            salmon_soft = workbook.add_format({"bg_color": "#FADBD8"})

            formats_by_col = {}
            for col in ["Requested Date", "Received Date", "Due Date"]:
                if col in sheet.columns:
                    formats_by_col[col] = date_format

            _autofit_widths(worksheet, sheet, formats_by_col=formats_by_col, min_w=9, max_w=42, pad=2, ratio=1.1)

            max_row, max_col = sheet.shape
            worksheet.add_table(
                0, 0, max_row, max_col - 1,
                {
                    "columns": [{"header": col} for col in sheet.columns],
                    "style": "Table Style Light 9"
                }
            )

            dias_col_idx = sheet.columns.get_loc(dias_col_name) if dias_col_name in sheet.columns else None
            money_fmt_cache = {}

            def get_money_format(bg_color=None, font_color=None):
                key = (bg_color, font_color)
                if key in money_fmt_cache:
                    return money_fmt_cache[key]
                base = dict(money_format_base)
                if bg_color:
                    base["bg_color"] = bg_color
                if font_color:
                    base["font_color"] = font_color
                fmt = workbook.add_format(base)
                money_fmt_cache[key] = fmt
                return fmt

            for row in range(1, max_row + 1):
                dias_val = sheet.iloc[row - 1, dias_col_idx] if dias_col_idx is not None else None
                row_bg = None
                row_font = None
                row_fmt = None

                if name == "Need":
                    if dias_val is not None and pd.notna(dias_val) and dias_val > 3:
                        row_fmt = yellow_fill
                        row_bg = "#FFFF00"
                elif name == "Flag_File":
                    row_fmt = red_fill_white
                    row_bg = "#FF0000"
                    row_font = "#FFFFFF"
                elif name == "Empty_or_Number":
                    if dias_val is not None and pd.notna(dias_val) and dias_val >= 3:
                        row_fmt = yellow_fill
                        row_bg = "#FFFF00"
                    else:
                        row_fmt = salmon_soft
                        row_bg = "#FADBD8"

                for col in range(len(sheet.columns)):
                    col_name = sheet.columns[col]
                    value = sheet.iloc[row - 1, col]

                    if col_name == "Total Amount":
                        value_num = None
                        if pd.notna(value):
                            try:
                                value_num = float(value)
                            except Exception:
                                try:
                                    value_num = float(re.sub(r"[^\d\.\-]", "", str(value)))
                                except Exception:
                                    value_num = None

                        money_fmt = get_money_format(bg_color=row_bg, font_color=row_font)
                        if value_num is None or pd.isna(value_num):
                            worksheet.write(row, col, "", money_fmt if row_bg else row_fmt)
                        else:
                            worksheet.write_number(row, col, float(value_num), money_fmt)
                    else:
                        fmt_to_use = row_fmt
                        if pd.isna(value):
                            worksheet.write(row, col, "", fmt_to_use)
                        else:
                            worksheet.write(row, col, value, fmt_to_use)

    xlsx_buffer.seek(0)

    st.success("✅ Processing completed.")

    if preview:
        with st.expander("Preview sheets"):
            for name, sheet in groups.items():
                st.markdown(f"**{name}** – {len(sheet)} rows")
                st.dataframe(sheet.head(200))

    st.download_button(
        label=f"⬇️ Download {output_filename}",
        data=xlsx_buffer.getvalue(),
        file_name=output_filename,
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
    )

st.caption("Tip: if Supabase is not available, the 'Flag_File' sheet will not include the 'Sales Rep' column.")
