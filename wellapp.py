import streamlit as st
import math
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import io
import PyPDF2
from scipy.interpolate import interp1d
from scipy.integrate import odeint

st.set_page_config(page_title="Advanced Rod Pump Calculator for Gas Wells", layout="wide")

# API 11L Rod Pump Data
API_11L_PUMPS = {
    "25-175-RWAC-20-4": {"plunger_diameter": 1.75, "stroke_length": 20},
    "30-185-RHBC-20-4": {"plunger_diameter": 1.85, "stroke_length": 20},
    "40-200-RWAC-24-4": {"plunger_diameter": 2.00, "stroke_length": 24},
    "80-275-RHAC-40-5": {"plunger_diameter": 2.75, "stroke_length": 40},
    "114-285-RHAC-64-6": {"plunger_diameter": 2.85, "stroke_length": 64},
    "160-300-RHAC-100-6": {"plunger_diameter": 3.00, "stroke_length": 100},
    "228-375-RHAC-144-8": {"plunger_diameter": 3.75, "stroke_length": 144},
}

# Rod Data as per API RP 11L
ROD_DATA = {
    "5/8": {"diameter": 0.625, "weight": 0.749, "stretch": 0.878, "strength": 9.91},
    "3/4": {"diameter": 0.750, "weight": 1.082, "stretch": 0.608, "strength": 14.3},
    "7/8": {"diameter": 0.875, "weight": 1.471, "stretch": 0.446, "strength": 19.5},
    "1": {"diameter": 1.000, "weight": 1.918, "stretch": 0.341, "strength": 25.5},
    "1 1/8": {"diameter": 1.125, "weight": 2.423, "stretch": 0.269, "strength": 32.3},
}

# Fluid properties
FLUID_PROPERTIES = {
    "Water": {"specific_gravity": 1.0, "viscosity": 1.0},
    "Light Oil": {"specific_gravity": 0.85, "viscosity": 5.0},
    "Medium Oil": {"specific_gravity": 0.93, "viscosity": 50.0},
    "Heavy Oil": {"specific_gravity": 0.98, "viscosity": 500.0},
}

# Units
UNITS = {
    "US": {"length": "ft", "pressure": "psi", "weight": "lbs", "viscosity": "cP"},
    "Metric": {"length": "m", "pressure": "kPa", "weight": "kg", "viscosity": "mPa·s"},
}

# Conversion factors
CONVERSION_FACTORS = {
    "ft_to_m": 0.3048,
    "m_to_ft": 3.28084,
    "psi_to_kPa": 6.89476,
    "kPa_to_psi": 0.145038,
    "lbs_to_kg": 0.453592,
    "kg_to_lbs": 2.20462,
}

# Global variables to store calculation results
global_results = {}

# Calculation functions
@st.cache_data
def calculate_s(c, d1, d2):
    if d1 == 0:
        raise ValueError("Beam dimension 1 (d1) cannot be zero.")
    return 2 * c * d2 / d1

@st.cache_data
def calculate_n(l, s, c, h):
    if s == 0 or h == 0 or c == h:
        raise ValueError("Invalid input: s, h cannot be zero, and c cannot equal h.")
    return math.sqrt(70471.2 * l / (s * (1 - c/h)))

@st.cache_data
def calculate_ap(dp):
    return math.pi * dp**2 / 4

@st.cache_data
def calculate_ar(dr):
    return math.pi * dr**2 / 4

@st.cache_data
def calculate_wf(sf, d, ap):
    return sf * 62.4 * d * ap / 144

@st.cache_data
def calculate_wr(y, d, ar):
    return y * d * ar / 144

@st.cache_data
def calculate_f1(s, n, c, h):
    if h == 0:
        raise ValueError("h cannot be zero.")
    return s * n**2 * (1 + c/h) / 70471.2

@st.cache_data
def calculate_prl_max(wf, sf, wr, wf_f1, y):
    if y == 0:
        raise ValueError("y cannot be zero.")
    return wf - sf * 62.4 * wr / y + wr + wf_f1

@st.cache_data
def calculate_t(s, wf, n, wr):
    return 0.25 * s * (wf + 2 * s * n**2 * wr / 70471.2)

@st.cache_data
def calculate_f2(s, n, c, h):
    if h == 0:
        raise ValueError("h cannot be zero.")
    return s * n**2 * (1 - c/h) / 70471.2

@st.cache_data
def calculate_prl_min(sf, wr, y, wf_f2):
    if y == 0:
        raise ValueError("y cannot be zero.")
    return -sf * 62.4 * wr / y + wr - wf_f2

@st.cache_data
def calculate_c(prl_max, prl_min):
    return 0.5 * (prl_max + prl_min)

@st.cache_data
def calculate_rod_stress(prl_max, prl_min, ar):
    if ar == 0:
        raise ValueError("Total rod area cannot be zero.")
    return (prl_max - prl_min) / (2 * ar)

@st.cache_data
def perform_calculations(d, dp, d1, d2, c, c_h_ratio, l, rod_sections):
    try:
        s = calculate_s(c, d1, d2)
        h = c / c_h_ratio if c_h_ratio != 0 else 0
        n = calculate_n(l, s, c, h)
        ap = calculate_ap(dp)
        
        ar_total = sum(calculate_ar(ROD_DATA[dr]["diameter"]) * lr for dr, lr in rod_sections if lr > 0)
        ar = ar_total / d if d != 0 else 0
        
        if ar == 0:
            raise ValueError("Total rod area cannot be zero. Please check rod sections.")
        
        sf = 1.0  # Specific gravity of fluid (placeholder)
        y = 490  # Specific weight of steel (lb/ft^3)
        
        wf = calculate_wf(sf, d, ap)
        wr = calculate_wr(y, d, ar)
        f1 = calculate_f1(s, n, c, h)
        prl_max = calculate_prl_max(wf, sf, wr, wf*f1, y)
        t = calculate_t(s, wf, n, wr)
        f2 = calculate_f2(s, n, c, h)
        prl_min = calculate_prl_min(sf, wr, y, wf*f2)
        counterbalance = calculate_c(prl_max, prl_min)
        rod_stress = calculate_rod_stress(prl_max, prl_min, ar)

        return s, n, ap, ar, wf, wr, f1, prl_max, t, f2, prl_min, counterbalance, rod_stress
    except ValueError as e:
        raise ValueError(f"Calculation error: {str(e)}")
    except Exception as e:
        raise Exception(f"An unexpected error occurred: {str(e)}")

@st.cache_data
def calculate_spm(stroke_length, plunger_diameter, desired_rate):
    """Calculate Strokes Per Minute (SPM) based on desired production rate."""
    plunger_area = math.pi * (plunger_diameter / 2) ** 2
    volume_per_stroke = plunger_area * stroke_length / 1728  # Convert to ft³
    spm = desired_rate / (volume_per_stroke * 1440)  # Convert to SPM
    return spm

@st.cache_data
def calculate_dog_leg_severity(md1, inc1, azi1, md2, inc2, azi2):
    """Calculate dog leg severity between two survey points."""
    dls = np.arccos(np.cos(np.radians(inc2 - inc1)) -
                    np.sin(np.radians(inc1)) * np.sin(np.radians(inc2)) *
                    (1 - np.cos(np.radians(azi2 - azi1)))) * (180 / np.pi) * 100 / (md2 - md1)
    return dls

@st.cache_data
def process_deviation_data(deviation_df):
    """Process deviation data and calculate additional parameters."""
    deviation_df['Inc_rad'] = np.radians(deviation_df['Angle'])
    deviation_df['Azi_rad'] = np.radians(deviation_df['Azimuth'])
    
    deviation_df['DLS'] = 0.0
    for i in range(1, len(deviation_df)):
        deviation_df.loc[i, 'DLS'] = calculate_dog_leg_severity(
            deviation_df.loc[i-1, 'SD'], deviation_df.loc[i-1, 'Angle'], deviation_df.loc[i-1, 'Azimuth'],
            deviation_df.loc[i, 'SD'], deviation_df.loc[i, 'Angle'], deviation_df.loc[i, 'Azimuth']
        )
    
    return deviation_df

@st.cache_data
def extract_deviation_data_from_pdf(uploaded_file):
    """Extract deviation data from the uploaded PDF file."""
    try:
        pdf_reader = PyPDF2.PdfReader(uploaded_file)
        text = ""
        for page in pdf_reader.pages:
            text += page.extract_text() + "\n"
        
        lines = text.split('\n')
        data = [line.split() for line in lines if line.strip() and line.split()[0].isdigit()]
        df = pd.DataFrame(data)

        if len(df.columns) >= 10:
            df.columns = ['Sl No', 'SD', 'Angle', 'Azimuth', 'TVD', 'N-S', 'E-W', 'Net Drift', 'Net Dir', 'VS'] + list(df.columns[10:])
        else:
            st.error("The extracted data does not have the expected number of columns. Please check the PDF format.")
            return None

        numeric_columns = ['SD', 'Angle', 'Azimuth', 'TVD', 'N-S', 'E-W', 'Net Drift', 'Net Dir', 'VS']
        for col in numeric_columns:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')

        return df
    except Exception as e:
        st.error(f"An error occurred while processing the PDF: {str(e)}")
        return None

@st.cache_data
def plot_wellpath(deviation_df):
    """Plot 3D well path."""
    fig = go.Figure(data=[go.Scatter3d(
        x=deviation_df['E-W'],
        y=deviation_df['N-S'],
        z=deviation_df['TVD'],
        mode='lines',
        line=dict(color='blue', width=2)
    )])
    
    fig.update_layout(
        scene=dict(
            xaxis_title='East-West (ft)',
            yaxis_title='North-South (ft)',
            zaxis_title='TVD (ft)',
            aspectmode='manual',
            aspectratio=dict(x=1, y=1, z=0.5)
        ),
        title='3D Well Path'
    )
    
    return fig

def calculate_rod_stress_profile(rod_sections, prl_max, well_depth):
    """Calculate stress profile along the rod string."""
    depths = [0]
    stresses = [prl_max / rod_sections[0]['area']]
    current_depth = 0
    current_load = prl_max
    
    for section in rod_sections:
        current_depth += section['length']
        current_load -= section['weight']
        depths.append(current_depth)
        stresses.append(current_load / section['area'])
    
    return depths, stresses

def optimize_rod_string(well_depth, prl_max, fluid_load):
    """Optimize rod string based on calculated stresses."""
    available_sizes = list(ROD_DATA.keys())
    optimal_rod_string = []
    remaining_depth = well_depth
    current_load = prl_max

    while remaining_depth > 0:
        optimal_size = available_sizes[0]
        for size in available_sizes:
            rod_data = ROD_DATA[size]
            stress = current_load / (np.pi * (rod_data['diameter'] / 2) ** 2)
            if stress < rod_data['strength'] * 0.8:  # Using 80% of yield strength as safety factor
                optimal_size = size
                break
        
        section_length = min(remaining_depth, 1000)  # Max 1000 ft per section
        optimal_rod_string.append({
            "size": optimal_size,
            "length": section_length,
            "diameter": ROD_DATA[optimal_size]["diameter"],
            "weight": ROD_DATA[optimal_size]["weight"] * section_length,
            "area": np.pi * (ROD_DATA[optimal_size]["diameter"] / 2) ** 2
        })
        
        remaining_depth -= section_length
        current_load -= ROD_DATA[optimal_size]["weight"] * section_length

    return optimal_rod_string



@st.cache_data
def calculate_torque_and_power(prl_max, prl_min, stroke_length, spm, crank_radius, rod_weight):
    """
    Calculate torque and power for rod pumps in gas wells.
    
    Parameters:
    prl_max (float): Maximum polished rod load (lbs)
    prl_min (float): Minimum polished rod load (lbs)
    stroke_length (float): Stroke length (inches)
    spm (float): Strokes per minute
    crank_radius (float): Crank radius (inches)
    rod_weight (float): Total weight of the rod string (lbs)
    
    Returns:
    dict: A dictionary containing calculated torque and power values
    """
    # Convert units
    stroke_length_ft = stroke_length / 12
    crank_radius_ft = crank_radius / 12
    
    # Calculate angular velocity
    omega = 2 * math.pi * spm / 60  # rad/sec
    
    # Calculate maximum and minimum torque
    T_max = prl_max * crank_radius_ft
    T_min = prl_min * crank_radius_ft
    
    # Calculate average torque
    T_avg = (T_max + T_min) / 2
    
    # Calculate power
    power_ft_lbs_sec = T_avg * omega
    power_hp = power_ft_lbs_sec / 550
    
    # Calculate gearbox torque
    gearbox_efficiency = 0.95  # Assumed efficiency
    gearbox_torque = T_max / gearbox_efficiency
    
    # Calculate counterbalance effect
    counterbalance_effect = rod_weight * stroke_length_ft / (2 * math.pi)
    
    return {
        "max_torque": T_max,
        "min_torque": T_min,
        "avg_torque": T_avg,
        "power_hp": power_hp,
        "gearbox_torque": gearbox_torque,
        "counterbalance_effect": counterbalance_effect
    }

# (Keep all the existing data structures)

# Modify the calculate_dyna_card function for faster calculation using 3D Gibbs equation

@st.cache_data

def calculate_dyna_card(surface_position, surface_load, well_depth, rod_string, fluid_load, tubing_pressure, casing_pressure, plunger_area, gas_gravity, damping_coefficient, friction_coefficient):
    """Generate accurate dyna card for gas wells using 3D Gibbs equations with optimized performance."""
    try:
        st.write("Debug: Entering calculate_dyna_card function")
        E = 30e6  # Young's modulus for steel (psi)
        g = 32.2  # Acceleration due to gravity (ft/s^2)
        
        total_length = sum(section['length'] for section in rod_string)
        total_mass = sum(section['weight'] for section in rod_string) / g
        avg_density = total_mass / total_length
        avg_area = sum(section['area'] * section['length'] for section in rod_string) / total_length

        def gibbs_equations(y, t, params):
            x, v = y
            E, A, rho, L, F_surface, damping, friction = params
            stress = F_surface / A
            strain = x / L
            a = (E * A / (rho * L)) * (stress - E * strain) - damping * v - friction * np.sign(v)
            return [v, a]

        # Ensure all inputs are numpy arrays
        surface_position = np.array(surface_position)
        surface_load = np.array(surface_load)
        t = np.linspace(0, 1, len(surface_position))

        # Initialize arrays for results
        downhole_positions = np.zeros_like(surface_position)
        downhole_loads = np.zeros_like(surface_load)

        # Solve ODE for each point
        for i in range(len(surface_position)):
            params = [E, avg_area, avg_density, total_length, surface_load[i], damping_coefficient, friction_coefficient]
            y0 = [0, 0]  # Initial displacement and velocity
            sol = odeint(gibbs_equations, y0, [0, 1], args=(params,))
            displacement = sol[-1, 0]
            downhole_positions[i] = surface_position[i] - displacement
            downhole_loads[i] = surface_load[i] - total_mass * g + fluid_load

        # Ensure downhole position is always positive
        min_pos = np.min(downhole_positions)
        downhole_positions -= min_pos

        st.write("Debug: Exiting calculate_dyna_card function successfully")
        return downhole_positions.tolist(), np.maximum(downhole_loads, 0).tolist()
    except Exception as e:
        st.error(f"Error in calculate_dyna_card: {str(e)}")
        st.write(f"Debug: Exception occurred - {str(e)}")
        return [], []


# Implement Vogel's IPR for gas wells
def calculate_vogel_ipr(reservoir_pressure, pwf_range, aof):
    """Calculate Inflow Performance Relationship (IPR) using Vogel's correlation for gas wells."""
    q = aof * (1 - 0.2 * (pwf_range/reservoir_pressure) - 0.8 * (pwf_range/reservoir_pressure)**2)
    return q

# Modify the main function
def main():
    st.title("Advanced Rod Pump Calculator for Gas Wells")

    # Add unit selection
    unit_system = st.sidebar.selectbox("Select Unit System", list(UNITS.keys()))

    tab1, tab2, tab3, tab4, tab5 = st.tabs(["Rod Pump Design", "Well Path Analysis", "Dyna Card", "Power and Torque", "IPR Analysis"])

    with tab1:
        st.header("Rod Pump Design Calculator")

        col1, col2 = st.columns(2)

        with col1:
            st.subheader("Input Parameters")
            
            # Well Parameters
            st.write("Well Parameters")
            well_depth = st.number_input(f"Well Depth ({UNITS[unit_system]['length']})", value=3000.0)
            tubing_pressure = st.number_input(f"Tubing Pressure ({UNITS[unit_system]['pressure']})", value=100.0)
            casing_pressure = st.number_input(f"Casing Pressure ({UNITS[unit_system]['pressure']})", value=50.0)
            
            # Fluid Selection
            selected_fluid = st.selectbox("Select Fluid Type", list(FLUID_PROPERTIES.keys()))
            fluid_specific_gravity = FLUID_PROPERTIES[selected_fluid]["specific_gravity"]
            fluid_viscosity = FLUID_PROPERTIES[selected_fluid]["viscosity"]
            st.write(f"Fluid Specific Gravity: {fluid_specific_gravity}")
            st.write(f"Fluid Viscosity: {fluid_viscosity} {UNITS[unit_system]['viscosity']}")
            
            # Additional Well Parameters
            bottom_hole_temperature = st.number_input("Bottom Hole Temperature (°F)", value=180.0)
            water_cut = st.slider("Water Cut (%)", 0, 100, 20)
            gas_oil_ratio = st.number_input("Gas-Oil Ratio (scf/bbl)", value=500.0)
            
            # Pump Selection
            st.write("Pump Selection")
            selected_pump = st.selectbox("Select API 11L Pump", list(API_11L_PUMPS.keys()))
            plunger_diameter = API_11L_PUMPS[selected_pump]["plunger_diameter"]
            max_stroke_length = API_11L_PUMPS[selected_pump]["stroke_length"]
            
            st.write(f"Plunger diameter: {plunger_diameter} in")
            stroke_length = st.slider("Stroke Length (in)", min_value=1, max_value=max_stroke_length, value=max_stroke_length)
            
            # Production Rate
            desired_rate = st.number_input("Desired Production Rate (bbl/day)", value=100.0)
            
            # Calculate SPM
            spm = calculate_spm(stroke_length, plunger_diameter, desired_rate)
            st.write(f"Calculated SPM: {spm:.2f}")
            
            # Surface Unit Parameters
            st.write("Surface Unit Parameters")
            d1 = st.number_input("Beam dimension 1 (in)", value=96.05)
            d2 = st.number_input("Beam dimension 2 (in)", value=111.0)
            c = st.number_input("Crank length (in)", value=37.0)
            c_h_ratio = st.number_input("Crank to pitman ratio", value=0.33)
            l = st.number_input("Max allowable acceleration factor", value=0.4)
            
            # Rod String Design
            st.subheader("Rod String Design")
            rod_sections = []
            total_length = 0
            for i in range(4):
                col_a, col_b = st.columns(2)
                with col_a:
                    rod_size = st.selectbox(f"Section {i+1} Size", list(ROD_DATA.keys()), key=f"rod_size_{i}")
                with col_b:
                    length = st.number_input(f"Section {i+1} Length (ft)", value=0.0, key=f"length_{i}")
                    total_length += length
                rod_sections.append({
                    "size": rod_size,
                    "length": length,
                    "diameter": ROD_DATA[rod_size]["diameter"],
                    "weight": ROD_DATA[rod_size]["weight"] * length,
                    "area": math.pi * (ROD_DATA[rod_size]["diameter"] / 2) ** 2
                })
            
            if total_length != well_depth:
                st.warning(f"Total rod length ({total_length} ft) does not match well depth ({well_depth} ft)")

        with col2:
            st.subheader("Results")

            try:
                # Perform calculations
                plunger_area = math.pi * (plunger_diameter / 2) ** 2
                fluid_load = (tubing_pressure - casing_pressure) * plunger_area * 144  # Convert psi to psf
                
                s, n, ap, ar, wf, wr, f1, prl_max, t, f2, prl_min, counterbalance, rod_stress = perform_calculations(
                    well_depth, plunger_diameter, d1, d2, c, c_h_ratio, l, [(section['size'], section['length']) for section in rod_sections])

                # Store results in global_results
                global_results.update({
                    'well_depth': well_depth,
                    'stroke_length': stroke_length,
                    'prl_min': prl_min,
                    'prl_max': prl_max,
                    'rod_sections': rod_sections,
                    'fluid_load': fluid_load,
                    'pumping_speed': spm,
                    'plunger_area': plunger_area
                })

                # Calculate torque and power
                torque_power_results = calculate_torque_and_power(prl_max, prl_min, stroke_length, spm, c/2, wr)

                results = {
                    "Parameter": ["Stroke length (S)", "Pumping speed (N)", "Plunger area (Ap)", "Rod area (Ar)",
                                  "Fluid load (Wf)", "Rod weight (Wr)", "F1", "Max polished rod load (PRL_max)",
                                  "Peak torque (T)", "F2", "Min polished rod load (PRL_min)", "Counterbalance (C)",
                                  "Rod stress", "Max Torque", "Min Torque", "Avg Torque", "Power", "Gearbox Torque"],
                    "Value": [s, spm, ap, ar, wf, wr, f1, prl_max, t, f2, prl_min, counterbalance, rod_stress,
                              torque_power_results["max_torque"], torque_power_results["min_torque"],
                              torque_power_results["avg_torque"], torque_power_results["power_hp"],
                              torque_power_results["gearbox_torque"]],
                    "Unit": ["in", "SPM", "in²", "in²", "lbs", "lbs", "", "lbs", "in-lbs", "", "lbs", "lbs", "psi",
                             "ft-lbs", "ft-lbs", "ft-lbs", "HP", "ft-lbs"]
                }
                
                df = pd.DataFrame(results)
                df['Value'] = df['Value'].round(2)
                st.table(df)

                # Polished Rod Load vs Position Chart
                positions = np.linspace(0, s, 100)
                prl = [prl_min + (prl_max - prl_min) * (1 - np.cos(2 * np.pi * pos / s)) / 2 for pos in positions]
                    
                prl_chart = go.Figure(data=[go.Scatter(x=positions, y=prl, mode='lines', name='Polished Rod Load')])
                prl_chart.update_layout(
                    title='Polished Rod Load vs Position',
                    xaxis_title='Position (in)',
                    yaxis_title='Polished Rod Load (lbs)',
                    showlegend=True
                )
                st.plotly_chart(prl_chart, use_container_width=True)

                # Rod Stress Profile Chart
                depths, stresses = calculate_rod_stress_profile(rod_sections, prl_max, well_depth)
                stress_profile_chart = go.Figure(data=[go.Scatter(x=depths, y=stresses, mode='lines', name='Rod Stress')])
                stress_profile_chart.update_layout(
                    title='Rod Stress Profile',
                    xaxis_title='Depth (ft)',
                    yaxis_title='Stress (psi)',
                    showlegend=True
                )
                st.plotly_chart(stress_profile_chart, use_container_width=True)

            except ValueError as e:
                st.error(f"Calculation error: {str(e)}")
            except Exception as e:
                st.error(f"An unexpected error occurred: {str(e)}")

    with tab2:
        st.header("Well Path Analysis")
        
        uploaded_file = st.file_uploader("Choose a PDF file with deviation data", type="pdf")
        
        if uploaded_file is not None:
            deviation_data = extract_deviation_data_from_pdf(uploaded_file)
            
            if deviation_data is not None:
                st.subheader("Deviation Data")
                st.dataframe(deviation_data)
                
                processed_data = process_deviation_data(deviation_data)
                
                st.subheader("Dog Leg Severity")
                dls_chart = go.Figure(data=[go.Scatter(
                    x=processed_data['SD'],
                    y=processed_data['DLS'],
                    mode='lines+markers',
                    name='Dog Leg Severity'
                )])
                dls_chart.update_layout(
                    title='Dog Leg Severity vs Measured Depth',
                    xaxis_title='Measured Depth (ft)',
                    yaxis_title='Dog Leg Severity (°/100ft)',
                    showlegend=True
                )
                st.plotly_chart(dls_chart, use_container_width=True)
                
                st.subheader("Well Path Visualization")
                wellpath_fig = plot_wellpath(processed_data)
                st.plotly_chart(wellpath_fig, use_container_width=True)
                
                # 2D Well Path Projections
                fig_2d = go.Figure()
                fig_2d.add_trace(go.Scatter(x=processed_data['E-W'], y=processed_data['N-S'], mode='lines', name='Plan View'))
                fig_2d.add_trace(go.Scatter(x=processed_data['E-W'], y=processed_data['TVD'], mode='lines', name='E-W Section'))
                fig_2d.add_trace(go.Scatter(x=processed_data['N-S'], y=processed_data['TVD'], mode='lines', name='N-S Section'))
                fig_2d.update_layout(
                    title='2D Well Path Projections',
                    scene=dict(
                        xaxis_title='East-West (ft)',
                        yaxis_title='North-South (ft) / TVD (ft)',
                    ),
                    showlegend=True
                )
                st.plotly_chart(fig_2d, use_container_width=True)
            else:
                st.warning("No valid deviation data found in the uploaded PDF.")
        else:
            st.info("Please upload a PDF file containing deviation data.")

    
    with tab3:
        st.header("Dyna Card")
    
        # Additional inputs for gas well dyna card
        gas_gravity = st.number_input("Gas Gravity (relative to air)", value=0.65, min_value=0.1, max_value=2.0)
        damping_coefficient = st.number_input("Damping Coefficient", value=0.1, min_value=0.0, max_value=1.0)
        friction_coefficient = st.number_input("Friction Coefficient", value=0.2, min_value=0.0, max_value=1.0)

        if st.button("Generate Dyna Card"):
            st.write("Debug: Generate Dyna Card button clicked")
            if all(key in global_results for key in ['prl_min', 'prl_max', 'stroke_length', 'well_depth', 'rod_sections', 'fluid_load', 'plunger_area']):
                st.write("Debug: All required parameters found in global_results")
                try:
                    surface_positions = np.linspace(0, global_results['stroke_length'], 100)
                    surface_loads = global_results['prl_min'] + (global_results['prl_max'] - global_results['prl_min']) * (1 - np.cos(2 * np.pi * surface_positions / global_results['stroke_length'])) / 2
                    
                    st.write(f"Debug: Surface positions range: {surface_positions.min():.2f} to {surface_positions.max():.2f}")
                    st.write(f"Debug: Surface loads range: {surface_loads.min():.2f} to {surface_loads.max():.2f}")
                    
                    with st.spinner("Generating Dyna Card..."):
                        downhole_positions, downhole_loads = calculate_dyna_card(
                            surface_positions.tolist(),
                            surface_loads.tolist(),
                            global_results['well_depth'],
                            global_results['rod_sections'],
                            global_results['fluid_load'],
                            tubing_pressure,
                            casing_pressure,
                            global_results['plunger_area'],
                            gas_gravity,
                            damping_coefficient,
                            friction_coefficient
                        )
                    
                    downhole_positions = np.array(downhole_positions)
                    downhole_loads = np.array(downhole_loads)
                    
                    st.write(f"Debug: Downhole positions range: {downhole_positions.min():.2f} to {downhole_positions.max():.2f}")
                    st.write(f"Debug: Downhole loads range: {downhole_loads.min():.2f} to {downhole_loads.max():.2f}")
                    
                    if len(downhole_positions) > 0 and len(downhole_loads) > 0:
                        st.write("Debug: Successfully calculated dyna card")
                        
                        # Create Dyna Card plot
                        dyna_card = go.Figure()
                        dyna_card.add_trace(go.Scatter(x=surface_positions, y=surface_loads, mode='lines', name='Surface Card'))
                        dyna_card.add_trace(go.Scatter(x=downhole_positions, y=downhole_loads, mode='lines', name='Downhole Card'))
                        dyna_card.update_layout(
                            title='Dyna Card (Surface and Downhole)',
                            xaxis_title='Position (in)',
                            yaxis_title='Load (lbs)',
                            showlegend=True
                        )
                        st.plotly_chart(dyna_card, use_container_width=True)

                        # Efficiency calculation
                        surface_work = np.trapz(surface_loads, surface_positions)
                        downhole_work = np.trapz(downhole_loads, downhole_positions)
                        
                        st.write(f"Debug: Surface work calculation - {len(surface_loads)} loads, {len(surface_positions)} positions")
                        st.write(f"Debug: Downhole work calculation - {len(downhole_loads)} loads, {len(downhole_positions)} positions")
                        
                        if surface_work > 0:
                            pump_efficiency = (downhole_work / surface_work) * 100
                        else:
                            pump_efficiency = 0

                        st.write(f"Estimated Pump Efficiency: {pump_efficiency:.2f}%")

                        # Additional analysis
                        st.write(f"Surface Work: {surface_work:.2f} in-lbs")
                        st.write(f"Downhole Work: {downhole_work:.2f} in-lbs")
                        st.write(f"Max Surface Load: {max(surface_loads):.2f} lbs")
                        st.write(f"Min Surface Load: {min(surface_loads):.2f} lbs")
                        st.write(f"Max Downhole Load: {max(downhole_loads):.2f} lbs")
                        st.write(f"Min Downhole Load: {min(downhole_loads):.2f} lbs")
                        
                        # If downhole work is still zero, try an alternative calculation
                        if downhole_work == 0:
                            st.write("Debug: Downhole work is zero. Attempting alternative calculation.")
                            # Ensure downhole positions are in ascending order
                            sorted_indices = np.argsort(downhole_positions)
                            sorted_positions = downhole_positions[sorted_indices]
                            sorted_loads = downhole_loads[sorted_indices]
                            alternative_downhole_work = np.trapz(sorted_loads, sorted_positions)
                            st.write(f"Alternative Downhole Work: {alternative_downhole_work:.2f} in-lbs")
                    else:
                        st.error("Failed to generate Dyna Card. The calculation returned empty results.")
                        st.write("Debug: calculate_dyna_card returned empty lists")
                except Exception as e:
                    st.error(f"An error occurred while generating the Dyna Card: {str(e)}")
                    st.write(f"Debug: Exception details - {type(e).__name__}: {str(e)}")
            else:
                missing_params = [key for key in ['prl_min', 'prl_max', 'stroke_length', 'well_depth', 'rod_sections', 'fluid_load', 'plunger_area'] if key not in global_results]
                st.error(f"Please complete the Rod Pump Design calculations first. Missing parameters: {', '.join(missing_params)}")
    
    
    with tab4:
        st.header("Power and Torque Analysis")
    
        if 'prl_max' in global_results and 'prl_min' in global_results:
            # Calculate torque and power results
            torque_power_results = calculate_torque_and_power(
                global_results['prl_max'], 
                global_results['prl_min'], 
                global_results['stroke_length'], 
                global_results['pumping_speed'], 
                c / 2,  # Assuming 'c' is the crank length defined earlier
                sum(section['weight'] for section in global_results['rod_sections'])  # Total rod weight
            )
            
            power = torque_power_results["power_hp"]
            peak_torque = torque_power_results["max_torque"]
            
            st.write(f"Estimated Power: {power:.2f} HP")
            st.write(f"Peak Torque: {peak_torque:.2f} ft-lbs")
            
            # Add power and torque limits
            max_allowable_power = st.number_input("Maximum Allowable Power (HP)", value=50.0)
            max_allowable_torque = st.number_input("Maximum Allowable Torque (ft-lbs)", value=10000.0)
            
            if power > max_allowable_power:
                st.warning(f"Calculated power ({power:.2f} HP) exceeds maximum allowable power ({max_allowable_power} HP)")
            if peak_torque > max_allowable_torque:
                st.warning(f"Calculated peak torque ({peak_torque:.2f} ft-lbs) exceeds maximum allowable torque ({max_allowable_torque} ft-lbs)")
            
            # Power and Torque vs. Pumping Speed Chart
            speeds = np.linspace(1, 20, 100)
            powers = []
            torques = []
            for speed in speeds:
                results = calculate_torque_and_power(
                    global_results['prl_max'], 
                    global_results['prl_min'], 
                    global_results['stroke_length'], 
                    speed, 
                    c / 2, 
                    sum(section['weight'] for section in global_results['rod_sections'])
                )
                powers.append(results["power_hp"])
                torques.append(results["max_torque"])
            
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=speeds, y=powers, mode='lines', name='Power (HP)'))
            fig.add_trace(go.Scatter(x=speeds, y=torques, mode='lines', name='Peak Torque (ft-lbs)', yaxis='y2'))
            fig.update_layout(
                title='Power and Torque vs. Pumping Speed',
                xaxis_title='Pumping Speed (SPM)',
                yaxis_title='Power (HP)',
                yaxis2=dict(title='Peak Torque (ft-lbs)', overlaying='y', side='right'),
                legend=dict(x=0.7, y=1)
            )
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.error("Please complete the Rod Pump Design calculations first.")


    with tab5:
        st.header("IPR Analysis (Vogel's Method for Gas Wells)")

        reservoir_pressure = st.number_input("Reservoir Pressure (psi)", value=3000.0)
        aof = st.number_input("Absolute Open Flow (AOF) Rate (Mscf/day)", value=5000.0)

        pwf_range = np.linspace(0, reservoir_pressure, 100)
        q = calculate_vogel_ipr(reservoir_pressure, pwf_range, aof)

        ipr_chart = go.Figure(data=[go.Scatter(x=q, y=pwf_range, mode='lines', name='IPR Curve')])
        ipr_chart.update_layout(
            title='Inflow Performance Relationship (IPR) - Vogel\'s Method',
            xaxis_title='Flow Rate (Mscf/day)',
            yaxis_title='Flowing Bottomhole Pressure (psi)',
            showlegend=True
        )
        st.plotly_chart(ipr_chart, use_container_width=True)

        if 'pumping_speed' in global_results:
            pump_rate = global_results['pumping_speed'] * global_results['plunger_area'] * global_results['stroke_length'] / 1728 * 1440  # bbl/day
            gas_rate = pump_rate * gas_oil_ratio  # Mscf/day
            st.write(f"Current Gas Production Rate: {gas_rate:.2f} Mscf/day")
            
            # Find the intersection point
            pwf_pump = np.interp(gas_rate, q, pwf_range)
            st.write(f"Estimated Flowing Bottomhole Pressure: {pwf_pump:.2f} psi")

            # Add pump rate line to the chart
            ipr_chart.add_trace(go.Scatter(x=[gas_rate, gas_rate], y=[0, pwf_pump], mode='lines', name='Current Rate'))
            ipr_chart.add_trace(go.Scatter(x=[0, gas_rate], y=[pwf_pump, pwf_pump], mode='lines', name='Pwf'))
            st.plotly_chart(ipr_chart, use_container_width=True)
        else:
            st.warning("Complete the Rod Pump Design calculations to see the current operating point on the IPR curve.")

    st.sidebar.title("About")
    st.sidebar.info("This Advanced Rod Pump Calculator for Gas Wells helps in determining various parameters for sucker rod pump design "
                    "based on API 11L standards, provides well path analysis based on deviation data, generates Dyna Cards, "
                    "and performs power, torque, and IPR analysis.")

    st.sidebar.title("Instructions")
    st.sidebar.markdown("""
    1. Use the "Rod Pump Design" tab to calculate pump parameters:
       - Input well parameters (depth, pressures, fluid properties)
       - Select an API 11L pump and set the stroke length
       - Specify rod sections using API standard sizes
    2. Use the "Well Path Analysis" tab to upload deviation data and visualize the well path
    3. Check the "Dyna Card" tab to view surface and downhole cards, accounting for rod stretch, compression, and friction
    4. Analyze power and torque requirements in the "Power and Torque" tab
    5. Perform IPR analysis in the "IPR Analysis" tab
    6. Analyze the results and visualizations to optimize your rod pump design for gas wells
    """)

if __name__ == "__main__":
    main()
   
