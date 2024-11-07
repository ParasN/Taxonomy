import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta

# Set page config
st.set_page_config(
    page_title="NextWave Taxonomy Dashboard",
    page_icon="📊",
    layout="wide"
)

# Custom CSS
st.markdown("""
    <style>
    .metric-card {
        background-color: #f8f9fa;
        padding: 20px;
        border-radius: 10px;
        box-shadow: 2px 2px 10px rgba(0,0,0,0.1);
    }
    .phase-header {
        font-size: 20px;
        font-weight: bold;
        margin-bottom: 10px;
    }
    </style>
""", unsafe_allow_html=True)

def create_project_overview():
    st.header("Project Overview")
    
    # Create three columns
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.subheader("Overall Progress")
        
        # Overall progress gauge
        phases = {
            "Phase 1 (POC: Evaluate multiple models)": 90,
            "Phase 2 (Finetuning Shortlisted Models)": 25,
            "Phase 3 (Implementation)": 0,
            "Phase 4 (Productionising)": 0
        }
        overall_progress = sum([prog for prog in phases.values()]) / len(phases)
        
        fig = go.Figure(go.Indicator(
            mode="gauge+number",
            value=overall_progress,
            domain={'x': [0, 1], 'y': [0, 1]},
            title={'text': "Completion %"},
            gauge={
                'axis': {'range': [None, 100]},
                'bar': {'color': "#1f77b4"},
                'steps': [
                    {'range': [0, 33], 'color': "#fee8c8"},
                    {'range': [33, 66], 'color': "#fdbb84"},
                    {'range': [66, 100], 'color': "#e6550d"}
                ]
            }
        ))
        fig.update_layout(height=300)
        st.plotly_chart(fig, use_container_width=True)
        
        # Phase-wise progress
        for phase, progress in phases.items():
            st.markdown(f"**{phase}**: {progress}%")
            st.progress(progress/100)
    
    with col3:
        st.subheader("Completed Tasks")
        completed_tasks = [
            {
                "task": "FashionClip Evaluation",
                "completion_date": "2024-10-10",
                "impact": "High"
            },
            {
                "task": "Llama3.2 Evaluation",
                "completion_date": "2024-10-20",
                "impact": "High"
            },
            {
                "task": "MiniCPM Evaluation",
                "completion_date": "2024-11-01",
                "impact": "Medium"
            },
            {
                "task": "Hybrid Approach Evaluation",
                "completion_date": "2024-11-01",
                "impact": "Medium"
            }
        ]
        
        for task in completed_tasks:
            with st.expander(f"✅ {task['task']}"):
                st.markdown(f"""
                    **Completed:** {task['completion_date']}  
                    **Impact:** {task['impact']}
                """)
    
    with col2:
        st.subheader("In Progress")
        current_tasks = [
            {
                "task": "Finetuning FashionCLip",
                "deadline": "2024-11-13",
                "progress": 25,
                "status": "On Track",
                "owner": "Couture"
            }
        ]
        
        for task in current_tasks:
            with st.expander(f"🔄 {task['task']}"):
                st.progress(task['progress']/100)
                st.markdown(f"""
                    **Progress:** {task['progress']}%  
                    **Deadline:** {task['deadline']}  
                    **Status:** {task['status']}  
                    **Owner:** {task['owner']}
                """)
                
                # Add status indicator
                if task['status'] == "On Track":
                    st.markdown("🟢 On Track")
                elif task['status'] == "Needs Attention":
                    st.markdown("🟡 Needs Attention")
                else:
                    st.markdown("🔴 Delayed")

    st.markdown("""
    <style>
    .stExpander {
        background-color: #f8f9fa;
        border-radius: 8px;
        margin-bottom: 10px;
    }
    .status-indicator {
        font-weight: bold;
        padding: 4px 8px;
        border-radius: 4px;
    }
    </style>
    """, unsafe_allow_html=True)


    # Timeline
    st.subheader("Project Timeline")
    
    # Create sample timeline data
    timeline_data = pd.DataFrame({
        'Task': ['POC - Model Evaluation', 'Finetuning', 'Data Integration Pipeline', 'Testing & Validation', 'Deployment'],
        'Start': pd.to_datetime(['2024-10-01', '2024-11-15', '2024-12-01', '2024-12-15', '2025-01-01']),
        'End': pd.to_datetime(['2024-11-15', '2024-11-30', '2024-12-15', '2024-12-31', '2025-01-31']),
        'Progress': [90, 50, 0, 0, 0]
    })
    
    fig = px.timeline(timeline_data, x_start="Start", x_end="End", y="Task", color="Progress",
                     color_continuous_scale="Viridis")
    fig.update_layout(height=250)
    st.plotly_chart(fig, use_container_width=True)

def create_model_evaluation():
    st.header("Model Evaluation Results")
    
    # Create tabs for different views
    tab1, tab2 = st.tabs(["Performance Comparison", "Detailed Metrics"])
    
    with tab1:
        col1, col2 = st.columns(2)
        
        with col1:
            # Using actual benchmark data from CSV
            performance_data = {
                'Model': ['FashionCLIP', 'Llama 3.2', 'MiniCPM'],
                'Direct Prediction': [40.4, 45.3, 48.7],  # Average f1_score for direct prediction
                'Top-K Prediction': [56.2, 59.3, 60.0]    # Average f1_score for top-k prediction
            }
            
            performance_df = pd.DataFrame(performance_data)
            
            # Create stacked horizontal bar chart
            fig = go.Figure()
            
            # Add bars for direct prediction
            fig.add_trace(go.Bar(
                y=performance_df['Model'],
                x=performance_df['Direct Prediction'],
                name='Direct Prediction',
                orientation='h',
                marker=dict(color='#1f77b4'),
                text=performance_df['Direct Prediction'].apply(lambda x: f'{x:.1f}%'),
                textposition='auto',
            ))
            
            # Add bars for additional top-k accuracy
            fig.add_trace(go.Bar(
                y=performance_df['Model'],
                x=performance_df['Top-K Prediction'] - performance_df['Direct Prediction'],
                name='Top-K Additional',
                orientation='h',
                marker=dict(color='#2ca02c'),
                text=(performance_df['Top-K Prediction'] - performance_df['Direct Prediction']).apply(lambda x: f'+{x:.1f}%'),
                textposition='auto',
            ))
            
            fig.update_layout(
                barmode='stack',
                height=400,
                title='Model Performance Comparison (F1-Score)',
                xaxis_title='F1-Score (%)',
                xaxis=dict(range=[0, 100]),
                showlegend=True,
                legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
            )
            st.plotly_chart(fig, use_container_width=True)

            st.info("""
            **Note on Performance Comparison:**
            - MiniCPM shows best overall performance with 45.3% F1-score
            - MiniCPM performs better on color attributes (39.8% F1-score)
            - FashionCLIP has more consistent performance across attributes
            """)
        
            # Add detailed performance analysis
            st.success("""
            **Key Performance Insights:**
            
            **Strengths by Model:**
            - **Llama 3.2**: Excels in sleeve detection (75.6% F1-score) and overall accuracy
            - **MiniCPM**: Best performance on color attributes and high sleeve detection accuracy
            - **FashionCLIP**: Most consistent across attributes with lower variance
            """)
            
            # Add recommendations based on results
            st.warning("""
            **Recommendations:**
            - Use Llama 3.2 for general attribute extraction
            - Consider MiniCPM for color-specific tasks
            - FashionCLIP for cases requiring consistent performance
            - Fine-tune models on challenging attributes (neckline, pattern)
            - Collect more diverse training data for weaker attributes
            - Investigate hybrid approaches combining model strengths
            """)
        
        with col2:
            # Attribute-wise accuracy using actual benchmark data
            attributes = ['color', 'pattern', 'neckline', 'sleeve']
            models = ['FashionCLIP', 'Llama 3.2', 'MiniCPM']
            
            # Average f1_scores for each attribute and model (from benchmark data)
            attribute_data = pd.DataFrame({
                'Attribute': attributes * 3,
                'Model': ['FashionCLIP'] * 4 + ['Llama 3.2'] * 4 + ['MiniCPM'] * 4,
                'F1-Score': [
                    # FashionCLIP
                    34.9, 19.2, 14.8, 10.4,
                    # Llama 3.2
                    39.4, 35.3, 21.5, 75.6,
                    # MiniCPM
                    39.8, 35.0, 18.0, 84.2
                ]
            })
            
            # Create grouped bar chart
            fig = px.bar(attribute_data, 
                        x='F1-Score', 
                        y='Attribute',
                        color='Model',
                        barmode='group',
                        orientation='h',
                        text=attribute_data['F1-Score'].apply(lambda x: f'{x:.1f}%'),
                        title='Attribute-wise F1-Score by Model',
                        color_discrete_sequence=['#1f77b4', '#ff7f0e', '#2ca02c'])
            
            fig.update_layout(
                height=400,
                xaxis_title='F1-Score (%)',
                xaxis=dict(range=[0, 100]),
                showlegend=True,
                legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
            )
            st.plotly_chart(fig, use_container_width=True)
            
            # Add explanatory note
            st.info("""
            **Note on Metrics:**
            - Direct Prediction: Model's performance on first prediction
            - Top-K Prediction: Model's performance when considering top k predictions
            - F1-Score combines both precision and recall for a balanced metric
            """)
    
    with tab2:
        add_detailed_metrics_tab()
        
def create_confusion_matrix_plot(confusion_matrix, labels, title):
    fig = go.Figure(data=go.Heatmap(
        z=confusion_matrix,
        x=labels,
        y=labels,
        colorscale='Viridis',
        texttemplate="%{z}%",
        textfont={"size": 14},
        hoverongaps=False,
    ))
    
    fig.update_layout(
        title=title,
        xaxis_title="Predicted",
        yaxis_title="Actual",
        height=500,
        width=600,
        xaxis={'side': 'bottom'}
    )
    
    return fig

def add_detailed_metrics_tab():
    st.subheader("Confusion Matrices")
    
    # Define attributes
    attributes = ['Color', 'Pattern', 'Neckline', 'Sleeve']
    
    # Sample confusion matrix data for each model
    confusion_matrices = {
        'FashionCLIP': np.array([
            [92, 5, 2, 1],
            [4, 88, 6, 2],
            [3, 7, 85, 5],
            [2, 3, 4, 91]
        ]),
        'Llama 3.2': np.array([
            [94, 4, 1, 1],
            [3, 90, 5, 2],
            [2, 6, 87, 5],
            [1, 2, 3, 94]
        ]),
        'MiniCPM': np.array([
            [95, 3, 1, 1],
            [2, 91, 5, 2],
            [2, 5, 88, 5],
            [1, 2, 3, 94]
        ])
    }
    
    # Model selection
    model_select = st.selectbox("Select Model", 
                              ["FashionCLIP", "Llama 3.2", "MiniCPM"])
    
    # Create two columns
    col1, col2 = st.columns([3, 2])
    
    with col1:
        # Display confusion matrix
        fig = create_confusion_matrix_plot(
            confusion_matrices[model_select],
            attributes,
            f"Confusion Matrix - {model_select}"
        )
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        # Calculate and display metrics
        cm = confusion_matrices[model_select]
        
        # Calculate overall accuracy
        accuracy = np.trace(cm) / np.sum(cm)
        
        # Calculate per-class metrics
        precision = np.diag(cm) / np.sum(cm, axis=0)
        recall = np.diag(cm) / np.sum(cm, axis=1)
        f1_score = 2 * (precision * recall) / (precision + recall)
        
        # Display summary metrics
        st.subheader("Summary Metrics")
        metrics_cols = st.columns(3)
        metrics_cols[0].metric(
            "Overall Accuracy",
            f"{accuracy*100:.1f}%",
            "Model Performance"
        )
        metrics_cols[1].metric(
            "Avg Precision",
            f"{np.mean(precision)*100:.1f}%",
            "Classification Precision"
        )
        metrics_cols[2].metric(
            "Avg Recall",
            f"{np.mean(recall)*100:.1f}%",
            "Classification Recall"
        )
        
        # Create and display detailed metrics table
        st.subheader("Detailed Metrics by Attribute")
        metrics_df = pd.DataFrame({
            'Attribute': attributes,
            'Precision (%)': precision * 100,
            'Recall (%)': recall * 100,
            'F1-Score (%)': f1_score * 100
        })
        metrics_df = metrics_df.round(2)
        
        # Apply custom styling to the dataframe
        st.dataframe(
            metrics_df.style.background_gradient(subset=['Precision (%)', 'Recall (%)', 'F1-Score (%)'],
                                               cmap='YlGn'),
            use_container_width=True
        )
    
    # Additional Analysis Section
    st.markdown("---")
    st.subheader("Model Performance Analysis")
    
    # Error Analysis
    error_cols = st.columns(2)
    
    with error_cols[0]:
        st.write("**Most Common Misclassifications:**")
        # Find top misclassifications
        misclass_data = []
        cm = confusion_matrices[model_select]
        for i in range(len(attributes)):
            for j in range(len(attributes)):
                if i != j:
                    misclass_data.append({
                        'Actual': attributes[i],
                        'Predicted': attributes[j],
                        'Occurance': cm[i][j]
                    })
        
        misclass_df = pd.DataFrame(misclass_data)
        misclass_df = misclass_df.sort_values('Occurance', ascending=False).head(3)
        st.dataframe(misclass_df, use_container_width=True)
    
    with error_cols[1]:
        st.write("**Best Performing Attributes:**")
        # Calculate per-attribute accuracy
        attr_accuracy = []
        for idx, attr in enumerate(attributes):
            accuracy = cm[idx][idx] / np.sum(cm[idx]) * 100
            attr_accuracy.append({
                'Attribute': attr,
                'Accuracy (%)': accuracy
            })
        
        accuracy_df = pd.DataFrame(attr_accuracy)
        accuracy_df = accuracy_df.sort_values('Accuracy (%)', ascending=False)
        st.dataframe(accuracy_df, use_container_width=True)
    
    # Performance Insights
    st.markdown("### Key Insights")
    col1, col2 = st.columns(2)
    
    with col1:
        st.info(f"""
        **Strengths:**
        - Overall accuracy of {accuracy:.1f}%
        - High precision in {attributes[np.argmax(precision)]} detection
        - Strong recall for {attributes[np.argmax(recall)]}
        """)
    
    with col2:
        st.warning(f"""
        **Areas for Improvement:**
        - Reduced accuracy in {attributes[np.argmin(precision)]} classification
        - Some confusion between {misclass_df.iloc[0]['Actual']} and {misclass_df.iloc[0]['Predicted']}
        - Potential for improved {attributes[np.argmin(recall)]} detection
        """)


# Main dashboard layout
st.title("📊 NextWave Taxonomy Project Dashboard")

# Create sections
create_project_overview()
st.markdown("---")
create_model_evaluation()

# Add timestamp
st.sidebar.markdown(f"*Last updated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}*")

# Add filters in sidebar
st.sidebar.header("Filters")
st.sidebar.date_input("Date Range Start", datetime.now() - timedelta(days=30))
st.sidebar.date_input("Date Range End", datetime.now())
st.sidebar.multiselect("Categories", ["Western Wear", "Ethnic Wear", "Kids Wear"])
st.sidebar.multiselect("Models", ["FashionCLIP", "Llama", "Hybrid"])

def create_next_steps():
    st.header("Next Steps & Action Items")
    
    # Create columns for different categories
    col1, col2 = st.columns(2)
    
    with col1:
        # Critical Tasks
        st.subheader("Critical Tasks")
        tasks = {
            "Evaluate finetuned FashionCLip Model": {
                "deadline": "2024-11-18",
                "status": "Awaiting Finetuning Completion",
                "priority": "High",
            },
            "Do Cost Analysis of GPU inferencing": {
                "deadline": "2024-11-15",
                "status": "In Progress",
                "priority": "High",
            },
            "Align Engineering for Productionising the Shortlisted Model": {
                "deadline": "2024-11-22",
                "status": "Pending",
                "priority": "Medium",
            }
        }
        
        for task, details in tasks.items():
            with st.expander(f"📌 {task}"):
                cols = st.columns(4)
                cols[0].markdown(f"**Deadline:** {details['deadline']}")
                cols[1].markdown(f"**Status:** {details['status']}")
                cols[2].markdown(f"**Priority:** {details['priority']}")
    
    with col2:
        # Risk Matrix
        st.subheader("Risk Assessment (Dummy)")
        risk_data = pd.DataFrame({
            'Risk': ['Data Quality Issues', 'Performance Degradation', 'Integration Delays'],
            'Impact': [8, 7, 6],
            'Probability': [6, 5, 7],
            'Status': ['Mitigating', 'Monitoring', 'Planning']
        })
        
        fig = px.scatter(risk_data, x='Impact', y='Probability', 
                        text='Risk', size='Impact',
                        color='Status', title='Risk Matrix')
        fig.update_layout(height=300)
        st.plotly_chart(fig, use_container_width=True)

def create_resource_utilization():
    st.header("Resource Utilization (Dummy)")
    
    # Create tabs for different resource views
    tab1, tab2, tab3 = st.tabs(["Compute Resources", "Memory Usage", "Cost Analysis"])
    
    with tab1:
        # Generate sample time series data for CPU usage
        dates = pd.date_range(start='2024-07-01', end='2024-07-31', freq='D')
        cpu_data = pd.DataFrame({
            'Date': dates,
            'CPU Usage (%)': np.random.normal(65, 15, len(dates)),
            'GPU Usage (%)': np.random.normal(75, 10, len(dates))
        })
        
        fig = px.line(cpu_data, x='Date', y=['CPU Usage (%)', 'GPU Usage (%)'],
                     title='Compute Resource Utilization')
        st.plotly_chart(fig, use_container_width=True)
        
        # Current Usage Metrics
        cols = st.columns(4)
        cols[0].metric("CPU Current", "68%", "2%")
        cols[1].metric("GPU Current", "75%", "-3%")
        cols[2].metric("Active Jobs", "12", "+2")
        cols[3].metric("Queue Length", "3", "-1")
    
    with tab2:
        # Memory usage breakdown
        memory_data = pd.DataFrame({
            'Component': ['Model Weights', 'Cache', 'Active Processing', 'System Reserved'],
            'Usage (GB)': [3.2, 1.8, 2.5, 0.5]
        })
        
        fig = px.pie(memory_data, values='Usage (GB)', names='Component',
                    title='Memory Usage Distribution')
        st.plotly_chart(fig, use_container_width=True)
        
        # Memory trends
        memory_trend = pd.DataFrame({
            'Date': dates,
            'Total Memory (GB)': np.random.normal(8, 1, len(dates)),
            'Available Memory (GB)': np.random.normal(4, 0.5, len(dates))
        })
        
        fig = px.area(memory_trend, x='Date', y=['Total Memory (GB)', 'Available Memory (GB)'],
                     title='Memory Usage Trends')
        st.plotly_chart(fig, use_container_width=True)
    
    with tab3:
        # Cost breakdown
        cost_data = pd.DataFrame({
            'Category': ['Compute', 'Storage', 'API Calls', 'Data Transfer'],
            'Cost ($)': [1200, 800, 500, 300],
            'Budget ($)': [1500, 1000, 600, 400]
        })
        
        fig = go.Figure(data=[
            go.Bar(name='Actual Cost', x=cost_data['Category'], y=cost_data['Cost ($)']),
            go.Bar(name='Budget', x=cost_data['Category'], y=cost_data['Budget ($)'])
        ])
        fig.update_layout(barmode='group', title='Cost vs Budget Analysis')
        st.plotly_chart(fig, use_container_width=True)

def create_documentation_updates():
    st.header("Documentation & Updates")
    st.subheader("Project Documentation")
        
    # Architecture Documentation
    with st.expander("🏗️ Project Documentation"):
            st.markdown("""
            * [System Design Overview](https://rilcloud-my.sharepoint.com/:i:/g/personal/paras_nagpal_ril_com/EQSuzRQppNVPv6JvACtAEWkBXef8Z0uFo2REwC2-8Zgj8w?e=vRB9vq)
            * [PRD](https://rilcloud-my.sharepoint.com/:w:/g/personal/paras_nagpal_ril_com/EdUH8SZ47VVAt6in1VcU3bkBNRhjKR2lEIy7udC51-75sA?e=eGggh5)
            * [POC Evaluation and Success Criteris](https://rilcloud-my.sharepoint.com/:w:/r/personal/paras_nagpal_ril_com/Documents/NextWave%20Taxonomy%20Requirements%20and%20Evaluation.docx?d=wd527825baa9b41aa98e7689f63789519&csf=1&web=1&e=aFH8ap)
            * [POC Evaluation Sheet](https://rilcloud-my.sharepoint.com/:x:/r/personal/shagun2_tyagi_ril_com/Documents/FTF%20IMpetus/Taxonomy/Taxonomy%20Model%20Benchmarking%20Analysis.xlsx?d=w71cbb1ebe03e451e949a8efedf714a53&csf=1&web=1&e=43sCg5)
            """)
        
    # User Guides
    with st.expander("📚 User Guides"):
            st.markdown("""
            * [End-to-End Guide](https://rilcloud-my.sharepoint.com/:w:/g/personal/paras_nagpal_ril_com/EdUH8SZ47VVAt6in1VcU3bkBNRhjKR2lEIy7udC51-75sA?e=eGggh5)
            * API Usage Examples
            * Best Practices
            """)
        
    # Known Issues
    with st.expander("⚠️ Known Issues"):
            issues = pd.DataFrame({
                'Issue': ['Memory leak in processing pipeline', 
                         'Occasional timeout in API responses',
                         'Inconsistent results for edge cases'],
                'Status': ['Investigating', 'Fixed', 'In Progress'],
                'Priority': ['High', 'Medium', 'Low'],
                'Reported': ['2024-07-25', '2024-07-20', '2024-07-15']
            })
            st.dataframe(issues)

# Add these functions to the main dashboard
st.markdown("---")
create_next_steps()
st.markdown("---")
create_resource_utilization()
st.markdown("---")
create_documentation_updates()

# Update sidebar with additional options
st.sidebar.markdown("---")
st.sidebar.subheader("View Options")
st.sidebar.checkbox("Show Resource Metrics", value=True)
st.sidebar.checkbox("Show Risk Matrix", value=True)
st.sidebar.checkbox("Show Known Issues", value=True)

# Add refresh button
if st.sidebar.button("Refresh Dashboard"):
    st.experimental_rerun()
