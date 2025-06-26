# --- Gather all fund metrics as before ---
rows = []
for name, code in mapping.items():
    if code in codes:
        nav = fetch_nav(code)
        m   = compute_metrics(nav)
        if m:
            m["Fund Name"] = name
            rows.append(m)

# --- Display ranking ---
if rows:
    df = pd.DataFrame(rows)

    # normalize + weighted score out of 10
    for metric, w in weights.items():
        vals = df[metric]
        if metric in ("SD", "Beta", "Downside Capture"):
            norm = (vals.max() - vals) / (vals.max() - vals.min())
        else:
            norm = (vals - vals.min()) / (vals.max() - vals.min())
        df[f"{metric}_score"] = norm * w

    df["Score"] = (df.filter(like="_score").sum(axis=1) * 10).round(2)

    # **FIXED**: build the correct list of columns
    display_cols = ["Fund Name"] + list(weights.keys()) + ["Score"]

    display = df[display_cols].round(2).sort_values("Score", ascending=False).reset_index(drop=True)
    display.index += 1
    display.insert(0, "SL No", display.index)

    st.dataframe(display, use_container_width=True)
else:
    st.info("🔔 Select at least one fund above to see metrics.")
