import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from algorithms import (
    detect_isolation_forest, 
    detect_density_lof, 
    detect_roc, 
    detect_iqr, 
    detect_multivariate_robust,
    apply_cascade_voting
)

# --- Helper Functions ---
def load_data(file):
    df = pd.read_csv(file)
    return df

# --- Main Application ---
def main():
    st.set_page_config(page_title="Golden Period Utility", layout="wide")

    st.title("Golden Period Selection Utility")
    st.caption("Univariate & Multivariate Fault Detection | Domain-Tuned Cascade | Coverage Reporting")
    st.markdown("---")

    # 1. Dataset Upload
    uploaded_file = st.file_uploader("Upload Time-Series Dataset (CSV)", type=['csv'])

    if uploaded_file is not None:
        if 'main_df' not in st.session_state or (
            'uploaded_file_name' in st.session_state and 
            st.session_state.uploaded_file_name != uploaded_file.name
        ):
            try:
                df = load_data(uploaded_file)
                timestamp_col = df.columns[0]
                df[timestamp_col] = pd.to_datetime(df[timestamp_col], errors='coerce')
                
                st.session_state['main_df'] = df
                st.session_state['timestamp_col'] = timestamp_col
                st.session_state['all_tags'] = df.columns[1:].tolist()
                st.session_state['uploaded_file_name'] = uploaded_file.name
                st.session_state['outliers_df'] = None 
                st.session_state['final_clean_df'] = None
                
            except Exception as e:
                st.error(f"Error processing file: {e}")
                return

        # Display Dataset Info
        st.success(f"Loaded: {st.session_state['main_df'].shape[0]} timestamps | {len(st.session_state['all_tags'])} tags")

        # 2. Preview Selection
        st.header("2. Data Preview")
        preview_tag = st.selectbox("Select Tag", st.session_state['all_tags'], key="preview_select")
        if preview_tag:
            fig_preview = px.line(st.session_state['main_df'], x=st.session_state['timestamp_col'], y=preview_tag)
            st.plotly_chart(fig_preview, use_container_width=True)

        st.markdown("---")

        # 3. Manual Data Cleaning
        st.header("3. Manual Point Removal")
        col1, col2 = st.columns([1, 3])
        with col1:
            clean_tag = st.selectbox("Select Tag for Cleaning", st.session_state['all_tags'], key="clean_select")
        
        if clean_tag:
            fig_clean = px.scatter(
                st.session_state['main_df'], 
                x=st.session_state['timestamp_col'], 
                y=clean_tag, 
                title=f"Manual Removal: {clean_tag}"
            )
            fig_clean.update_traces(mode='lines+markers')
            fig_clean.update_layout(dragmode='select') 

            selection = st.plotly_chart(fig_clean, use_container_width=True, on_select="rerun", key="clean_plot")

            selected_indices = []
            if selection and "selection" in selection and "points" in selection["selection"]:
                selected_points = selection["selection"]["points"]
                selected_indices = [p.get("point_index", p.get("point_number")) for p in selected_points]

            if selected_indices:
                st.warning(f"{len(selected_indices)} points selected.")
                if st.button("Delete Selected Points"):
                    timestamps_to_remove = st.session_state['main_df'].iloc[selected_indices][st.session_state['timestamp_col']]
                    st.session_state['main_df'] = st.session_state['main_df'][
                        ~st.session_state['main_df'][st.session_state['timestamp_col']].isin(timestamps_to_remove)
                    ].reset_index(drop=True)
                    st.success("Points removed.")
                    st.rerun()

        st.markdown("---")

        # ======================================================
        # 4. Domain-Tuned Cascade (Univariate & Multivariate)
        # ======================================================
        st.header("4. Domain-Tuned Cascade Detection")
        st.info("Construct a cascade of algorithms to improve result stability.")
        
        col_algo, col_params = st.columns([1, 1])
        
        with col_algo:
            st.subheader("Select Algorithms")
            use_iso = st.checkbox("Tree-Based (Isolation Forest)", value=True)
            use_lof = st.checkbox("Density-Based (LOF)", value=False)
            use_roc = st.checkbox("Rate of Change (ROC)", value=False)
            use_iqr = st.checkbox("Statistical (IQR)", value=False)
            use_multi = st.checkbox("Multivariate (Robust Covariance)", value=False, help="Checks correlations between tags")
            
        with col_params:
            st.subheader("Tuning Parameters")
            contam = st.slider("Contamination (Expected % Outliers)", 0.01, 0.20, 0.05, 0.01)
            iqr_factor = st.slider("IQR Factor (Sensitivity)", 0.5, 3.0, 1.5, 0.1)
        
        # Cascade Logic
        selected_methods_count = sum([use_iso, use_lof, use_roc, use_iqr, use_multi])
        
        if selected_methods_count > 0:
            st.subheader("Cascade Logic")
            
            # --- FIX STARTS HERE ---
            # Only show slider if we have more than 1 method to vote on
            if selected_methods_count > 1:
                voting_threshold = st.slider(
                    "Minimum Votes to Flag Anomaly", 
                    min_value=1, 
                    max_value=selected_methods_count, 
                    value=1,
                    help="1 = Aggressive (Union), Max = Conservative (Intersection)"
                )
            else:
                # If only 1 method is selected, threshold is automatically 1
                voting_threshold = 1
                st.caption(f"Single method selected. Voting threshold fixed at 1.")
            # --- FIX ENDS HERE ---

            if st.button("Run Cascade Analysis"):
                df = st.session_state['main_df']
                all_tags = st.session_state['all_tags']
                results = {} 

                progress_bar = st.progress(0)
                
                # Pre-calculate Multivariate if selected
                multi_mask = None
                if use_multi:
                    with st.spinner("Running Multivariate Analysis..."):
                        multi_mask = detect_multivariate_robust(df, contam)

                for i, tag in enumerate(all_tags):
                    series = df[tag]
                    tag_votes = {} 

                    # 1. Run Univariate Methods
                    if use_iso: tag_votes['ISO'] = detect_isolation_forest(series, contam)
                    if use_lof: tag_votes['LOF'] = detect_density_lof(series, contam)
                    if use_roc: tag_votes['ROC'] = detect_roc(series, contam)
                    if use_iqr: tag_votes['IQR'] = detect_iqr(series, iqr_factor)
                    
                    # 2. Add Multivariate Result 
                    if use_multi and multi_mask is not None:
                        tag_votes['MULTI'] = multi_mask.values

                    # 3. Apply Cascade (Voting)
                    final_tag_mask = apply_cascade_voting(tag_votes, voting_threshold)
                    results[tag] = final_tag_mask
                    
                    progress_bar.progress((i + 1) / len(all_tags))
                
                st.session_state['outliers_df'] = pd.DataFrame(results)
                st.session_state['cascade_config'] = f"Methods: {selected_methods_count} | Threshold: {voting_threshold}"
                st.success("Cascade Analysis Complete!")

        # ==========================================
        # 5. Visualization
        # ==========================================
        if st.session_state.get('outliers_df') is not None:
            st.header("5. Visual Validation")
            view_tag = st.selectbox("Select Tag to Inspect", st.session_state['all_tags'], key="inspect_select")
            
            df = st.session_state['main_df']
            mask = st.session_state['outliers_df'][view_tag]
            
            fig_out = go.Figure()
            fig_out.add_trace(go.Scatter(x=df[st.session_state['timestamp_col']], y=df[view_tag], mode='lines', name='Raw Data', line=dict(color='blue', width=1)))
            
            outliers = df[mask]
            if not outliers.empty:
                fig_out.add_trace(go.Scatter(x=outliers[st.session_state['timestamp_col']], y=outliers[view_tag], mode='markers', name='Cascade Outliers', marker=dict(color='red', symbol='x', size=6)))
                
            st.plotly_chart(fig_out, use_container_width=True)

        st.markdown("---")

        # ==========================================
        # 6. Report Generation
        # ==========================================
        st.header("6. Coverage & Anomaly Report")
        
        if st.session_state.get('outliers_df') is not None:
            if st.button("Generate Final Report"):
                outliers_df = st.session_state['outliers_df']
                df = st.session_state['main_df']
                
                # 1. Create Summary Statistics
                summary_data = []
                for tag in st.session_state['all_tags']:
                    total_pts = len(df)
                    n_outliers = outliers_df[tag].sum()
                    pct_outliers = (n_outliers / total_pts) * 100
                    
                    summary_data.append({
                        "Tag Name": tag,
                        "Total Points": total_pts,
                        "Outliers Detected": n_outliers,
                        "Outlier %": round(pct_outliers, 2),
                        "Status": "Stable" if pct_outliers < 10 else "High Variance"
                    })
                
                report_df = pd.DataFrame(summary_data)
                
                # 2. Display Report
                col_rep1, col_rep2 = st.columns([2, 1])
                with col_rep1:
                    st.dataframe(report_df, use_container_width=True)
                with col_rep2:
                    st.metric("Avg Outlier Reduction", f"{report_df['Outlier %'].mean():.2f}%")
                    st.metric("Tags Processed", len(report_df))

                # 3. Global Cleaning (Consolidated Golden Period)
                # Create a mask that is True if ANY tag has an outlier at that timestamp (Conservative cleaning)
                # Or based on user preference. Let's assume removing rows where ANY tag failed is safest for multivariate models.
                global_mask = outliers_df.any(axis=1)
                clean_df = df[~global_mask]
                
                st.subheader("Golden Period Extraction")
                st.write(f"Original Rows: {len(df)} -> Cleaned Rows: {len(clean_df)}")
                
                # Download Button
                csv = clean_df.to_csv(index=False).encode('utf-8')
                st.download_button(
                    label="Download Golden Period Dataset (CSV)",
                    data=csv,
                    file_name="golden_period_data.csv",
                    mime="text/csv"
                )

if __name__ == "__main__":
    main()