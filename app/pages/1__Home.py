import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from streamlit_echarts import st_echarts
import time
from datetime import datetime, timedelta
import json

# Page configuration
st.set_page_config(
    page_title="ECG Analysis Dashboard",
    page_icon="🫀",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for enhanced medical theme
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        background: linear-gradient(45deg, #2E8B57, #20B2AA);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        text-align: center;
        margin-bottom: 2rem;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.1);
    }
    
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1.5rem;
        border-radius: 15px;
        color: white;
        text-align: center;
        box-shadow: 0 8px 16px rgba(0, 0, 0, 0.2);
        margin-bottom: 1rem;
        transition: transform 0.3s ease;
    }
    
    .metric-card:hover {
        transform: translateY(-5px);
    }
    
    .alert-card {
        background: linear-gradient(135deg, #ff6b6b 0%, #ee5a52 100%);
        padding: 1.2rem;
        border-radius: 10px;
        color: white;
        margin: 0.8rem 0;
        border-left: 5px solid #ff4757;
        box-shadow: 0 4px 8px rgba(255, 75, 87, 0.3);
        animation: pulse 2s infinite;
    }
    
    .normal-card {
        background: linear-gradient(135deg, #2ed573 0%, #1e90ff 100%);
        padding: 1.2rem;
        border-radius: 10px;
        color: white;
        margin: 0.8rem 0;
        border-left: 5px solid #2ed573;
        box-shadow: 0 4px 8px rgba(46, 213, 115, 0.3);
    }
    
    .warning-card {
        background: linear-gradient(135deg, #ffa502 0%, #ff6348 100%);
        padding: 1.2rem;
        border-radius: 10px;
        color: white;
        margin: 0.8rem 0;
        border-left: 5px solid #ffa502;
        box-shadow: 0 4px 8px rgba(255, 165, 2, 0.3);
    }
    
    .interpretation-box {
        background: linear-gradient(135deg, #e8f5e8 0%, #d4edda 100%);
        padding: 2rem;
        border-radius: 15px;
        border-left: 8px solid #2E8B57;
        margin: 1.5rem 0;
        box-shadow: 0 4px 12px rgba(0,0,0,0.1);
        color: black;
    }
    
    .recommendation-item {
        background: linear-gradient(135deg, #e8f5e8 0%, #d4edda 100%);
        padding: 1rem;
        margin: 0.8rem 0;
        border-radius: 8px;
        border-left: 4px solid #28a745;
        box-shadow: 0 2px 6px rgba(40, 167, 69, 0.2);
        transition: transform 0.2s ease;
        color: black;
    }
    
    .recommendation-item:hover {
        transform: translateX(5px);
    }
    
    .control-panel {
        background: linear-gradient(135deg, #f1f3f4 0%, #ffffff 100%);
        padding: 1.5rem;
        border-radius: 15px;
        box-shadow: 0 4px 12px rgba(0,0,0,0.1);
        margin: 1rem 0;
    }
    
    .status-indicator {
        width: 12px;
        height: 12px;
        border-radius: 50%;
        display: inline-block;
        margin-right: 8px;
        animation: pulse 2s infinite;
    }
    
    .status-normal { background-color: #28a745; }
    .status-warning { background-color: #ffc107; }
    .status-critical { background-color: #dc3545; }
    
    @keyframes pulse {
        0% { opacity: 1; }
        50% { opacity: 0.7; }
        100% { opacity: 1; }
    }
    
    .info-panel {
        background: rgba(255, 255, 255, 0.9);
        backdrop-filter: blur(10px);
        padding: 1rem;
        border-radius: 10px;
        border: 1px solid rgba(255, 255, 255, 0.2);
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    }
    
    .stTabs [data-baseweb="tab-list"] {
        gap: 2px;
    }
    
    .stTabs [data-baseweb="tab"] {
        height: 50px;
        padding-left: 20px;
        padding-right: 20px;
        background-color: #f0f2f6;
        border-radius: 10px 10px 0 0;
        color: black;
    }
    
    .stTabs [aria-selected="true"] {
        background-color: #2E8B57;
        color: white;
    }
</style>
""", unsafe_allow_html=True)



def load_ecg_data(uploaded_file):
    """Load ECG data from uploaded file"""
    try:
        if uploaded_file.name.endswith('.csv'):
            df = pd.read_csv(uploaded_file)
        elif uploaded_file.name.endswith(('.xls', '.xlsx')):
            df = pd.read_excel(uploaded_file)
        else:
            st.error("Please upload a CSV or Excel file")
            return None, None, None, None
            
        # Try to identify time and ECG columns automatically
        time_col = None
        ecg_col = None
        
        # Look for common time column names
        for col in df.columns:
            if any(keyword in col.lower() for keyword in ['time', 't', 'timestamp', 'seconds', 'ms']):
                time_col = col
                break
                
        # Look for common ECG column names
        for col in df.columns:
            if any(keyword in col.lower() for keyword in ['ecg', 'ekg', 'signal', 'voltage', 'amplitude', 'lead']):
                ecg_col = col
                break
                
        # If not found, use first two columns
        if time_col is None:
            time_col = df.columns[0]
        if ecg_col is None:
            ecg_col = df.columns[1] if len(df.columns) > 1 else df.columns[0]
            
        return df[time_col].values, df[ecg_col].values, time_col, ecg_col
        
    except Exception as e:
        st.error(f"Error loading file: {str(e)}")
        return None, None, None, None



def analyze_ecg_data(time_data, ecg_data):
    """Analyze loaded ECG data and return medical insights"""
    
    if len(ecg_data) == 0:
        return None
        
    # Calculate basic metrics
    duration = time_data[-1] - time_data[0] if len(time_data) > 1 else 0
    sample_rate = len(time_data) / duration if duration > 0 else 1
    
    # Simple peak detection for heart rate estimation
    threshold = np.percentile(ecg_data, 85)
    peaks = []
    
    for i in range(1, len(ecg_data) - 1):
        if (ecg_data[i] > ecg_data[i-1] and 
            ecg_data[i] > ecg_data[i+1] and 
            ecg_data[i] > threshold):
            if not peaks or (time_data[i] - time_data[peaks[-1]]) > 0.4:
                peaks.append(i)
    
    # Calculate heart rate
    if len(peaks) > 1 and duration > 0:
        heart_rate = (len(peaks) - 1) * 60 / duration
    else:
        heart_rate = 0
    
    # Calculate heart rate variability
    rr_intervals = []
    if len(peaks) > 1:
        for i in range(1, len(peaks)):
            rr_interval = time_data[peaks[i]] - time_data[peaks[i-1]]
            rr_intervals.append(rr_interval)
    
    hrv = np.std(rr_intervals) * 1000 if rr_intervals else 0
    
    # Determine status and classifications
    rhythm_regularity = "Regular" if hrv < 50 else "Irregular"
    rhythm = "Normal Sinus Rhythm"
    alerts = []
    recommendations = []
    status = "normal"
    
    if heart_rate > 100:
        rhythm = "Sinus Tachycardia"
        status = "warning"
        alerts.append("Elevated heart rate detected (>100 BPM)")
        recommendations.extend([
            "Monitor for symptoms (palpitations, chest discomfort)",
            "Consider underlying causes (stress, caffeine, medications)",
            "Follow up with healthcare provider if persistent"
        ])
    elif heart_rate < 60:
        rhythm = "Sinus Bradycardia"
        status = "warning"
        alerts.append("Low heart rate detected (<60 BPM)")
        recommendations.extend([
            "Monitor for symptoms (dizziness, fatigue, syncope)",
            "Review medications that may affect heart rate",
            "Consider evaluation if symptomatic"
        ])
    
    if hrv > 100:
        rhythm = "Irregular Rhythm"
        status = "critical"
        alerts.append("Irregular heart rhythm detected")
        recommendations.extend([
            "Consider cardiology consultation",
            "Monitor for symptoms of arrhythmia",
            "Evaluate need for further cardiac monitoring"
        ])
    
    # Generate interpretation
    if not alerts:
        interpretation = f"ECG shows {rhythm.lower()} with heart rate of {heart_rate:.0f} BPM. No significant abnormalities detected in this recording."
        recommendations.extend([
            "Continue regular physical activity as tolerated",
            "Maintain heart-healthy lifestyle",
            "Regular follow-up as recommended by healthcare provider"
        ])
    else:
        interpretation = f"ECG shows {rhythm.lower()} with heart rate of {heart_rate:.0f} BPM. {' '.join(alerts)}"
    
    return {
        "heart_rate": int(heart_rate),
        "rhythm": rhythm,
        "rhythm_regularity": rhythm_regularity,
        "hrv": hrv,
        "interpretation": interpretation,
        "recommendations": recommendations,
        "alerts": alerts,
        "duration": duration,
        "sample_rate": sample_rate,
        "peaks": peaks,
        "status": status,
        "rr_intervals": rr_intervals
    }

def create_echarts_ecg(time_data, ecg_data, current_time=None, peaks=None):
    """Create ECG chart using ECharts"""
    
    # Prepare data for ECharts
    if current_time is not None:
        mask = time_data <= current_time
        display_time = time_data[mask]
        display_ecg = ecg_data[mask]
    else:
        display_time = time_data
        display_ecg = ecg_data
    
    # Convert to list of [x, y] pairs
    ecg_series_data = [[float(t), float(v)] for t, v in zip(display_time, display_ecg)]
    
    # Mark R-peaks if provided
    peak_data = []
    if peaks is not None and current_time is not None:
        for peak_idx in peaks:
            if peak_idx < len(time_data) and time_data[peak_idx] <= current_time:
                peak_data.append({
                    'coord': [float(time_data[peak_idx]), float(ecg_data[peak_idx])],
                    'itemStyle': {'color': '#ff4757'}
                })
    
    options = {
        "title": {
            "text": "ECG Signal Analysis",
            "left": "center",
            "textStyle": {"fontSize": 20, "color": "#2E8B57"}
        },
        "tooltip": {
            "trigger": "axis",
            "formatter": "Time: {b}s<br/>Amplitude: {c}mV"
        },
        "grid": {
            "left": "10%",
            "right": "10%",
            "bottom": "15%",
            "top": "15%"
        },
        "xAxis": {
            "type": "value",
            "name": "Time (seconds)",
            "nameLocation": "middle",
            "nameGap": 30,
            "axisLine": {"lineStyle": {"color": "#666"}},
            "splitLine": {"show": True, "lineStyle": {"color": "#e0e0e0"}}
        },
        "yAxis": {
            "type": "value",
            "name": "Amplitude (mV)",
            "nameLocation": "middle",
            "nameGap": 50,
            "axisLine": {"lineStyle": {"color": "#666"}},
            "splitLine": {"show": True, "lineStyle": {"color": "#e0e0e0"}}
        },
        "series": [
            {
                "name": "ECG",
                "type": "line",
                "data": ecg_series_data,
                "smooth": False,
                "symbol": "none",
                "lineStyle": {
                    "color": "#2E8B57",
                    "width": 2
                },
                "markPoint": {
                    "data": peak_data,
                    "symbol": "circle",
                    "symbolSize": 8
                } if peak_data else None
            }
        ],
        "animation": True,
        "animationDuration": 300
    }
    
    # Add cursor line for replay
    if current_time is not None:
        options["series"].append({
            "name": "Cursor",
            "type": "line",
            "data": [[current_time, min(display_ecg)], [current_time, max(display_ecg)]],
            "lineStyle": {
                "color": "#ff4757",
                "width": 3,
                "type": "dashed"
            },
            "symbol": "none"
        })
    
    return options

def create_heart_rate_chart(rr_intervals, heart_rates):
    """Create heart rate trend chart using ECharts"""
    
    if not rr_intervals:
        return {}
    
    # Calculate rolling heart rate
    time_points = list(range(len(heart_rates)))
    
    options = {
        "title": {
            "text": "Heart Rate Trend",
            "left": "center",
            "textStyle": {"fontSize": 16, "color": "#2E8B57"}
        },
        "tooltip": {"trigger": "axis"},
        "grid": {"left": "10%", "right": "10%", "bottom": "15%", "top": "15%"},
        "xAxis": {
            "type": "category",
            "data": [f"Beat {i+1}" for i in time_points],
            "name": "Beat Number"
        },
        "yAxis": {
            "type": "value",
            "name": "Heart Rate (BPM)",
            "min": 40,
            "max": 140
        },
        "series": [{
            "name": "Heart Rate",
            "type": "line",
            "data": heart_rates,
            "smooth": True,
            "lineStyle": {"color": "#667eea"},
            "areaStyle": {"color": "rgba(102, 126, 234, 0.2)"}
        }]
    }
    
    return options

def create_hrv_chart(rr_intervals):
    """Create HRV analysis chart using ECharts"""
    
    if not rr_intervals:
        return {}
    
    # Convert to milliseconds
    rr_ms = [interval * 1000 for interval in rr_intervals]
    
    options = {
        "title": {
            "text": "Heart Rate Variability",
            "left": "center",
            "textStyle": {"fontSize": 16, "color": "#2E8B57"}
        },
        "tooltip": {"trigger": "axis"},
        "grid": {"left": "10%", "right": "10%", "bottom": "15%", "top": "15%"},
        "xAxis": {
            "type": "category",
            "data": [f"RR{i+1}" for i in range(len(rr_ms))],
            "name": "RR Interval"
        },
        "yAxis": {
            "type": "value",
            "name": "Duration (ms)"
        },
        "series": [{
            "name": "RR Intervals",
            "type": "bar",
            "data": rr_ms,
            "itemStyle": {"color": "#20B2AA"}
        }]
    }
    
    return options

# Initialize session state
if 'ecg_data' not in st.session_state:
    st.session_state.ecg_data = None
    st.session_state.time_data = None
    st.session_state.is_playing = False
    st.session_state.current_time = 0
    st.session_state.analysis = None
    st.session_state.playback_speed = 1.0

# Main App Layout
st.markdown('<h1 class="main-header">🫀 Advanced ECG Analysis Dashboard</h1>', unsafe_allow_html=True)

# Sidebar for controls
with st.sidebar:
    st.header("📁 Data Input")
    
    # File upload with drag and drop
    uploaded_file = st.file_uploader(
        "Upload ECG Data",
        type=['csv', 'xlsx', 'xls'],
        help="Drag and drop or browse for CSV/Excel files containing ECG data"
    )
    
    if uploaded_file is not None:
        if st.button("📊 Load & Analyze ECG Data", type="primary", use_container_width=True):
            with st.spinner("Processing ECG data..."):
                progress_bar = st.progress(0)
                
                # Load data
                progress_bar.progress(25)
                time_data, ecg_data, time_col, ecg_col = load_ecg_data(uploaded_file)
                
                if time_data is not None and ecg_data is not None:
                    progress_bar.progress(50)
                    
                    # Analyze data
                    st.session_state.time_data = time_data
                    st.session_state.ecg_data = ecg_data
                    st.session_state.current_time = 0
                    
                    progress_bar.progress(75)
                    st.session_state.analysis = analyze_ecg_data(time_data, ecg_data)
                    
                    progress_bar.progress(100)
                    st.success(f"✅ Successfully loaded {len(ecg_data):,} data points")
                    
                    # Show file info
                    with st.expander("📄 File Information", expanded=True):
                        st.write(f"**Time Column:** {time_col}")
                        st.write(f"**ECG Column:** {ecg_col}")
                        st.write(f"**Duration:** {st.session_state.analysis['duration']:.2f}s")
                        st.write(f"**Sample Rate:** {st.session_state.analysis['sample_rate']:.1f} Hz")
    
    st.markdown("---")
    
    # Enhanced replay controls
    if st.session_state.ecg_data is not None:
        st.header("▶️ Playback Controls")
        
        max_time = st.session_state.time_data[-1] if len(st.session_state.time_data) > 0 else 10
        
        # Status indicator
        status_class = f"status-{st.session_state.analysis['status']}" if st.session_state.analysis else "status-normal"
        play_status = "Playing" if st.session_state.is_playing else "Paused"
        st.markdown(f'<span class="status-indicator {status_class}"></span><strong>{play_status}</strong>', unsafe_allow_html=True)
        
        # Time controls
        current_time = st.slider(
            "Timeline",
            0.0,
            float(max_time),
            float(st.session_state.current_time),
            step=0.1,
            help="Scrub through the ECG recording"
        )
        st.session_state.current_time = current_time
        
        # Speed control
        speed = st.select_slider(
            "Playback Speed",
            options=[0.1, 0.25, 0.5, 1.0, 2.0, 4.0, 8.0],
            value=st.session_state.playback_speed,
            format_func=lambda x: f"{x}x"
        )
        st.session_state.playback_speed = speed
        
        # Enhanced control buttons
        col1, col2 = st.columns(2)
        
        with col1:
            if st.button("⏮️", help="Reset to beginning", use_container_width=True):
                st.session_state.current_time = 0
                st.session_state.is_playing = False
                st.rerun()
            
            if st.button("⏭️", help="Jump to end", use_container_width=True):
                st.session_state.current_time = max_time
                st.session_state.is_playing = False
                st.rerun()
        
        with col2:
            play_button_text = "⏸️ Pause" if st.session_state.is_playing else "▶️ Play"
            if st.button(play_button_text, use_container_width=True):
                st.session_state.is_playing = not st.session_state.is_playing
                st.rerun()
        
        # Quick jump buttons
        st.write("**Quick Jump:**")
        jump_cols = st.columns(4)
        jump_times = [0.25, 0.5, 0.75, 1.0]
        jump_labels = ["25%", "50%", "75%", "100%"]
        
        for i, (col, time_pct, label) in enumerate(zip(jump_cols, jump_times, jump_labels)):
            with col:
                if st.button(label, key=f"jump_{i}", use_container_width=True):
                    st.session_state.current_time = max_time * time_pct
                    st.rerun()

# Main content area with tabs
if st.session_state.ecg_data is None:
    
    col1, col2, col3 = st.columns(3)
    
    with col1:

        st.markdown("""
        ### 📊 **Supported Formats**
        - CSV files (.csv)
        - Excel files (.xlsx, .xls)
        - Automatic column detection
        """)
    
    with col2:
        st.markdown("""
        ### 🔍 **Analysis Features**
        - Heart rate calculation
        - Rhythm classification
        - HRV analysis
        - Clinical recommendations
        """)
    
    with col3:
        st.markdown("""
        ### 🎮 **Interactive Controls**
        - Real-time playback
        - Speed adjustment
        - Timeline scrubbing
        - Peak detection visualization
        """)

else:
    # Auto-advance for playback
    if st.session_state.is_playing:
        max_time = st.session_state.time_data[-1]
        if st.session_state.current_time < max_time:
            st.session_state.current_time = min(
                st.session_state.current_time + 0.1 * st.session_state.playback_speed, 
                max_time
            )
            time.sleep(0.05)  # Faster refresh rate
            st.rerun()
        else:
            st.session_state.is_playing = False
            st.rerun()
    
    # Display metrics with enhanced cards
    if st.session_state.analysis:
        analysis = st.session_state.analysis
        
        col1, col2, col3, col4 = st.columns(4)

        with col1:
            st.markdown(f"""
            <div class="metric-card">
                <h3>❤️ Heart Rate</h3>
                <h1>{analysis['heart_rate']} BPM</h1>
                <small>{"Normal" if 60 <= analysis['heart_rate'] <= 100 else "Abnormal"}</small>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            st.markdown(f"""
            <div class="metric-card">
                <h3>🔄 Rhythm</h3>
                <h1>{analysis['rhythm_regularity']}</h1>
                <small>{analysis['rhythm']}</small>
            </div>
            """, unsafe_allow_html=True)
    
        with col3:
            st.markdown(f"""
            <div class="metric-card">
                <h3>⏱️ Duration</h3>
                <h1>{analysis['duration']:.1f}s</h1>
                <small>{len(analysis['peaks'])} beats detected</small>
            </div>
            """, unsafe_allow_html=True)
        
        with col4:
            hrv_status = "Low" if analysis['hrv'] < 30 else "Normal" if analysis['hrv'] < 50 else "High"
            st.markdown(f"""
            <div class="metric-card">
                <h3>📊 HRV</h3>
                <h1>{analysis['hrv']:.0f}ms</h1>
                <small>{hrv_status} variability</small>
            </div>
            """, unsafe_allow_html=True)
    
    # Tabbed interface for different views
    tab1, tab2, tab3, tab4 = st.tabs(["📈 ECG Waveform", "💓 Heart Rate Analysis", "📊 HRV Analysis", "🏥 Clinical Report"])

    with tab1:
        st.subheader("Real-time ECG Visualization")
        
        # Main ECG chart using ECharts
        if st.session_state.analysis:
            ecg_options = create_echarts_ecg(
                st.session_state.time_data, 
                st.session_state.ecg_data, 
                st.session_state.current_time,
                st.session_state.analysis['peaks']
            )
            st_echarts(options=ecg_options, height="500px", key="ecg_chart")
            
            # Current position info
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Current Time", f"{st.session_state.current_time:.1f}s")
            with col2:
                current_idx = int(st.session_state.current_time * st.session_state.analysis['sample_rate'])
                if current_idx < len(st.session_state.ecg_data):
                    st.metric("Current Amplitude", f"{st.session_state.ecg_data[current_idx]:.3f}mV")
                else:
                    st.metric("Current Amplitude", "N/A")
            with col3:
                progress = (st.session_state.current_time / st.session_state.analysis['duration']) * 100
                st.metric("Progress", f"{progress:.1f}%")
    
    with tab2:
        st.subheader("Heart Rate Trend Analysis")
        
        if st.session_state.analysis and st.session_state.analysis['rr_intervals']:
            # Calculate beat-to-beat heart rates
            rr_intervals = st.session_state.analysis['rr_intervals']
            heart_rates = [60/rr for rr in rr_intervals if rr > 0]
            
            if heart_rates:
                hr_options = create_heart_rate_chart(rr_intervals, heart_rates)
                st_echarts(options=hr_options, height="400px", key="hr_chart")
                
                # Statistics
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.metric("Average HR", f"{np.mean(heart_rates):.1f} BPM")
                with col2:
                    st.metric("Min HR", f"{np.min(heart_rates):.1f} BPM")
                with col3:
                    st.metric("Max HR", f"{np.max(heart_rates):.1f} BPM")
                with col4:
                    st.metric("HR Range", f"{np.max(heart_rates) - np.min(heart_rates):.1f} BPM")
        else:
            st.info("No heart rate data available for analysis.")
    
    with tab3:
        st.subheader("Heart Rate Variability Analysis")
        
        if st.session_state.analysis and st.session_state.analysis['rr_intervals']:
            rr_intervals = st.session_state.analysis['rr_intervals']
            
            hrv_options = create_hrv_chart(rr_intervals)
            st_echarts(options=hrv_options, height="400px", key="hrv_chart")
            
            # HRV Metrics
            col1, col2, col3 = st.columns(3)
            with col1:
                rmssd = np.sqrt(np.mean(np.diff(rr_intervals)**2)) * 1000
                st.metric("RMSSD", f"{rmssd:.1f} ms", help="Root Mean Square of Successive Differences")
            with col2:
                sdnn = np.std(rr_intervals) * 1000
                st.metric("SDNN", f"{sdnn:.1f} ms", help="Standard Deviation of NN intervals")
            with col3:
                pnn50 = (np.sum(np.abs(np.diff(rr_intervals)) > 0.05) / len(np.diff(rr_intervals))) * 100
                st.metric("pNN50", f"{pnn50:.1f}%", help="Percentage of successive RR intervals that differ by more than 50ms")
        else:
            st.info("No HRV data available for analysis.")
    
    with tab4:
        st.subheader("Clinical Analysis Report")
        
        if st.session_state.analysis:
            analysis = st.session_state.analysis
            
            # Status overview with enhanced styling
            status_icons = {
                "normal": "✅",
                "warning": "⚠️", 
                "critical": "🚨"
            }
            status_colors = {
                "normal": "normal-card",
                "warning": "warning-card",
                "critical": "alert-card"
            }
            
            st.markdown(f"""
            <div class="{status_colors[analysis['status']]}">
                <h3>{status_icons[analysis['status']]} Overall Status: {analysis['status'].title()}</h3>
            </div>
            """, unsafe_allow_html=True)
            
            # Clinical alerts
            if analysis['alerts']:
                st.markdown("### 🚨 Clinical Alerts")
                for i, alert in enumerate(analysis['alerts'], 1):
                    st.markdown(f"""
                    <div class="alert-card">
                        <strong>Alert {i}:</strong> {alert}
                    </div>
                    """, unsafe_allow_html=True)
            
            # Clinical interpretation
            st.markdown("### 🔍 Clinical Interpretation")
            st.markdown(f"""
            <div class="interpretation-box">
                <h4>Rhythm Classification: {analysis['rhythm']}</h4>
                <p><strong>Interpretation:</strong> {analysis['interpretation']}</p>
                <br>
                <h5>Key Findings:</h5>
                <ul>
                    <li><strong>Heart Rate:</strong> {analysis['heart_rate']} BPM</li>
                    <li><strong>Rhythm Regularity:</strong> {analysis['rhythm_regularity']}</li>
                    <li><strong>Heart Rate Variability:</strong> {analysis['hrv']:.1f} ms</li>
                    <li><strong>Recording Duration:</strong> {analysis['duration']:.2f} seconds</li>
                    <li><strong>R-Peaks Detected:</strong> {len(analysis['peaks'])}</li>
                </ul>
            </div>
            """, unsafe_allow_html=True)
            
            # Clinical recommendations
            st.markdown("### 💡 Clinical Recommendations")
            for i, recommendation in enumerate(analysis['recommendations'], 1):
                st.markdown(f"""
                <div class="recommendation-item">
                    <strong>{i}.</strong> {recommendation}
                </div>
                """, unsafe_allow_html=True)
            
            # Technical analysis details
            with st.expander("🔧 Technical Analysis Details", expanded=False):
                col1, col2 = st.columns(2)
                
                with col1:
                    st.markdown("**Signal Properties:**")
                    st.write(f"• Sample Rate: {analysis['sample_rate']:.1f} Hz")
                    st.write(f"• Total Samples: {len(st.session_state.ecg_data):,}")
                    st.write(f"• Signal Range: {np.min(st.session_state.ecg_data):.3f} to {np.max(st.session_state.ecg_data):.3f} mV")
                    st.write(f"• Signal Mean: {np.mean(st.session_state.ecg_data):.3f} mV")
                    st.write(f"• Signal STD: {np.std(st.session_state.ecg_data):.3f} mV")
                
                with col2:
                    st.markdown("**Cardiac Metrics:**")
                    if analysis['rr_intervals']:
                        avg_rr = np.mean(analysis['rr_intervals'])
                        st.write(f"• Average RR Interval: {avg_rr:.3f}s")
                        st.write(f"• RR Interval Range: {np.min(analysis['rr_intervals']):.3f} - {np.max(analysis['rr_intervals']):.3f}s")
                        st.write(f"• Coefficient of Variation: {(np.std(analysis['rr_intervals'])/avg_rr*100):.1f}%")
                    st.write(f"• Peak Detection Threshold: {np.percentile(st.session_state.ecg_data, 85):.3f} mV")
                
                # Show raw data preview
                st.markdown("**Raw Data Preview:**")
                preview_df = pd.DataFrame({
                    'Time (s)': st.session_state.time_data[:10],
                    'ECG (mV)': st.session_state.ecg_data[:10]
                })
                st.dataframe(preview_df, use_container_width=True)
            
            # Export options
            st.markdown("### 📤 Export Options")
            export_col1, export_col2, export_col3 = st.columns(3)
            
            with export_col1:
                # Generate report summary
                report_summary = f"""
ECG Analysis Report
Generated: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}

PATIENT DATA:
- Recording Duration: {analysis['duration']:.2f} seconds
- Sample Rate: {analysis['sample_rate']:.1f} Hz

CARDIAC METRICS:
- Heart Rate: {analysis['heart_rate']} BPM
- Rhythm: {analysis['rhythm']}
- Rhythm Regularity: {analysis['rhythm_regularity']}
- Heart Rate Variability: {analysis['hrv']:.1f} ms

CLINICAL INTERPRETATION:
{analysis['interpretation']}

ALERTS:
{chr(10).join(f"- {alert}" for alert in analysis['alerts']) if analysis['alerts'] else "No alerts"}

RECOMMENDATIONS:
{chr(10).join(f"{i}. {rec}" for i, rec in enumerate(analysis['recommendations'], 1))}
                """
                
                st.download_button(
                    label="📄 Download Report",
                    data=report_summary,
                    file_name=f"ecg_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt",
                    mime="text/plain"
                )
            
            with export_col2:
                # Create analysis data CSV
                analysis_data = pd.DataFrame({
                    'Time': st.session_state.time_data,
                    'ECG_Signal': st.session_state.ecg_data,
                    'Peak_Detected': [1 if i in analysis['peaks'] else 0 for i in range(len(st.session_state.ecg_data))]
                })
                
                csv = analysis_data.to_csv(index=False)
                st.download_button(
                    label="📊 Download Data",
                    data=csv,
                    file_name=f"ecg_data_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                    mime="text/csv"
                )
            
            with export_col3:
                # Generate JSON report for integration
                json_report = {
                    "timestamp": datetime.now().isoformat(),
                    "analysis": analysis,
                    "metadata": {
                        "total_samples": len(st.session_state.ecg_data),
                        "signal_range": [float(np.min(st.session_state.ecg_data)), float(np.max(st.session_state.ecg_data))],
                        "signal_statistics": {
                            "mean": float(np.mean(st.session_state.ecg_data)),
                            "std": float(np.std(st.session_state.ecg_data))
                        }
                    }
                }
                
                st.download_button(
                    label="🔗 Download JSON",
                    data=json.dumps(json_report, indent=2),
                    file_name=f"ecg_analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                    mime="application/json"
                )

# Real-time status bar at bottom
if st.session_state.ecg_data is not None:
    st.markdown("---")
    
    # Create a real-time status container
    status_container = st.container()
    
    with status_container:
        col1, col2, col3, col4, col5 = st.columns(5)
        
        with col1:
            if st.session_state.is_playing:
                st.markdown("🔴 **RECORDING**")
            else:
                st.markdown("⏸️ **PAUSED**")
        
        with col2:
            st.markdown(f"⏱️ **{st.session_state.current_time:.1f}s / {st.session_state.analysis['duration']:.1f}s**")
        
        with col3:
            st.markdown(f"💓 **{st.session_state.analysis['heart_rate']} BPM**")
        
        with col4:
            st.markdown(f"🎚️ **{st.session_state.playback_speed}x Speed**")
        
        with col5:
            status_emoji = {"normal": "✅", "warning": "⚠️", "critical": "🚨"}[st.session_state.analysis['status']]
            st.markdown(f"{status_emoji} **{st.session_state.analysis['status'].title()}**")

# Enhanced footer with medical disclaimer
st.markdown("---")
st.markdown("""
<div style='background: linear-gradient(135deg, #f8f9fa 0%, #e9ecef 100%); padding: 1.5rem; border-radius: 10px; text-align: center; margin-top: 2rem;'>
    <h4 style='color: #2E8B57; margin-bottom: 1rem;'>⚕️ Medical Disclaimer</h4>
    <p style='color: #666; margin-bottom: 0.5rem;'><strong>This dashboard is for educational and informational purposes only.</strong></p>
    <p style='color: #666; margin-bottom: 0.5rem;'>It should not be used as a substitute for professional medical advice, diagnosis, or treatment.</p>
    <p style='color: #666; margin-bottom: 1rem;'>Always consult with a qualified healthcare provider for medical decisions.</p>
    <div style='border-top: 1px solid #dee2e6; padding-top: 1rem; color: #999; font-size: 0.9rem;'>
        <p>ECG Analysis Dashboard v2.0 | Powered by Redbull & Coffee | © 2025</p>
    </div>
</div>
""", unsafe_allow_html=True)