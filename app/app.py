import io
import re
import numpy as np
import pandas as pd
import streamlit as st
from datetime import datetime

# =========================
# === SUPABASE HELPERS  ===
# =========================
try:
    from supabase import create_client
except Exception:
    create_client = None


@st.cache_data(ttl=300, show_spinner=False)
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
        st.error("❌ Supabase credentials missing in [supabase_sales] (need url, key).")
        st.write(cfg)
        return None

    # --- Try SDK path ---
    if create_client is not None:
        try:
            sb = create_client(url, key)
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
    else:
        st.info("Supabase SDK not installed. Falling back to REST API. Add 'supabase' to requirements.txt.")

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
# === UTILITIES          ===
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
    if s == "" or re.fullmatch(r"[A-Za-z0-9_\-/]+", s):
        return "empty_or_number"
    return "other"


def calculate_dias360(requested_date, ref_date):
    try:
        if pd.isna(requested_date):
            return None
        requested_date = pd.to_datetime(requested_date).date()
        diff = (ref_date.year - requested_date.year) * 360 + \
               (ref_date.month - requested_date.month) * 30 + \
               (ref_date.day - requested_date.day)
        return diff
    except Exception:
        return None


def coerce_money_series(s):
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


def build_salesrep_lookup(dfc: pd.DataFrame):
    dfc = dfc.copy()
    cols = {c.lower(): c for c in dfc.columns}
    source_col = cols.get("source")
    rep_col = cols.get("sales_rep") or cols.get("cus_sales_rep")

    if not source_col or not rep_col:
        raise ValueError("Missing required columns ('source', 'sales_rep' or 'cus_sales_rep').")

    dfc["po_key"] = dfc[source_col].apply(extract_digits)
    salesrep_map = (
        dfc.sort_values(by=["po_key"])
           .groupby("po_key", dropna=True)[rep_col]
           .first()
           .to_dict()
    )
    return {k: v for k, v in salesrep_map.items() if k}


# =========================
# === APP UI             ===
# =========================
st.set_page_config(page_title="FL Rev Confirmed Invoices", page_icon="📄")

st.title("📄 FL Rev Confirmed Invoices – Processor")
st.markdown((
    "Upload the **Expenses CSV** (AP > Expenses). "
    "The app will automatically fetch the **sales consolidated** data from **Supabase** "
    "(table `ventas_frutto`) to map the *Sales Rep*. "
    "If the connection or secret is missing, it will proceed without that column."
))

csv_file = st.file_uploader("Expenses CSV (ap-expenses-*.csv)", type=["csv"])
debug_supabase = st.checkbox("Show Supabase diagnostics", value=False)
preview = st.checkbox("Show data preview", value=False)

if st.button("Process", type="primary"):
    if csv_file is None:
        st.error("You must upload an Expenses CSV file.")
        st.stop()

    df = pd.read_csv(csv_file)
    df.columns = df.columns.str.strip()
    today = datetime.today().date()
    dias_col_name = today.strftime("%m/%d/%y")

    df["Total Amount"] = coerce_money_series(df["Total Amount"])
    for col in ["Requested Date", "Received Date", "Due Date"]:
        df[col] = pd.to_datetime(df[col], errors="coerce")

    desc_norm = df["Description"].astype(str).str.strip().str.lower()
    status_norm = df["Status"].astype(str).str.strip().str.lower()
    mask_produce = desc_norm == "produce"
    mask_paid_zero = (status_norm == "paid") & (df["Total Amount"].fillna(0) == 0)
    mask_unpaid_or_partial = status_norm.isin(["unpaid", "partially paid"])
    df = df[mask_produce & (mask_paid_zero | mask_unpaid_or_partial)].copy()

    df["Invoice Group"] = df["Invoice Number"].apply(classify_invoice)
    df[dias_col_name] = df["Requested Date"].apply(lambda x: calculate_dias360(x, today))
    df["Answer"] = ""

    for col in ["Requested Date", "Received Date", "Due Date"]:
        df[col] = df[col].dt.strftime("%m/%d/%y")

    sales_df = load_sales_with_diagnostics()
    salesrep_map = build_salesrep_lookup(sales_df) if sales_df is not None else None

    flag_mask_zero = df["Total Amount"].apply(lambda x: np.isclose(float(x or 0), 0.0))
    flag_mask_class = df["Invoice Group"] == "flag_file"
    flag_mask_pas = df["Invoice Number"].astype(str).str.contains("PAS", case=False, na=False)
    need_mask = df["Invoice Group"] == "need"
    flag_union = flag_mask_class | flag_mask_pas | (flag_mask_zero & ~need_mask)

    groups = {
        "Flag_File": df[flag_union].copy(),
        "Empty_or_Number": df[(df["Invoice Group"] == "empty_or_number") & (~flag_union)].copy(),
        "Need": df[(df["Invoice Group"] == "need") & (~flag_union)].copy(),
    }

    if salesrep_map and not groups["Flag_File"].empty:
        po_keys = groups["Flag_File"]["PO # / EXP #"].apply(extract_digits)
        groups["Flag_File"]["Sales Rep"] = po_keys.map(salesrep_map).fillna("")

    # --- Export ---
    output_name = f"FL REV CONFIRMED INVOICES {datetime.today().strftime('%B %d').upper()}.xlsx"
    buf = io.BytesIO()
    with pd.ExcelWriter(buf, engine="xlsxwriter") as writer:
        for name, sheet in groups.items():
            sheet.drop(columns=["Invoice Group"], errors="ignore").to_excel(writer, index=False, sheet_name=name)

    buf.seek(0)
    st.success("✅ File processed successfully.")
    st.download_button("⬇️ Download Excel", buf.getvalue(), file_name=output_name)
