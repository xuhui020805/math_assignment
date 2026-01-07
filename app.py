import streamlit as st
import numpy as np
import sympy as sp
import plotly.graph_objects as go
import plotly.figure_factory as ff

# --- 1. CONFIGURATION & TITLE ---
st.set_page_config(page_title="Gradient & Steepest Ascent Visualizer", layout="wide")

st.title("Topic 4: Gradient and Direction of Steepest Ascent")
st.markdown("""
This interactive application helps visualize **multivariable functions** and their **gradients**.
The gradient vector $\\nabla f(x,y)$ always points in the direction of steepest ascent.
""")

# --- 2. SIDEBAR INPUT ---
st.sidebar.header("Function Settings")

# Helper function to clean "Natural Math" input to "Python" input
def clean_input(equation):
    # Map standard superscripts to Python power syntax
    superscripts = {
        '⁰': '**0', '¹': '**1', '²': '**2', '³': '**3', '⁴': '**4',
        '⁵': '**5', '⁶': '**6', '⁷': '**7', '⁸': '**8', '⁹': '**9'
    }
    for sup, py_pow in superscripts.items():
        equation = equation.replace(sup, py_pow)
    
    # Handle implicit multiplication (optional but helpful, e.g., 2x -> 2*x)
    # Simple fix for basic cases, complex parsing handled by SymPy
    equation = equation.replace('^', '**') 
    return equation

# User Input
func_input = st.sidebar.text_input(
    "Enter a function f(x, y):", 
    value="x³ - y³ - 3x + 1", # Default value demonstrating x³ support
    help="You can use standard notation like x², x³, or Python notation x**2."
)

# Range Sliders
x_range = st.sidebar.slider("X Range", -5.0, 5.0, (-2.0, 2.0))
y_range = st.sidebar.slider("Y Range", -5.0, 5.0, (-2.0, 2.0))
resolution = st.sidebar.slider("Grid Resolution", 10, 50, 20)

# --- 3. MATHEMATICAL CALCULATION (SymPy) ---
try:
    x, y = sp.symbols('x y')
    
    # Preprocess the input string (Convert ³ to **3)
    cleaned_expr_str = clean_input(func_input)
    
    # Parse expression
    f_expr = sp.sympify(cleaned_expr_str)
    
    # Calculate Partial Derivatives
    fx_expr = sp.diff(f_expr, x)
    fy_expr = sp.diff(f_expr, y)
    
    # Create lambda functions for numerical evaluation
    f_func = sp.lambdify((x, y), f_expr, 'numpy')
    fx_func = sp.lambdify((x, y), fx_expr, 'numpy')
    fy_func = sp.lambdify((x, y), fy_expr, 'numpy')

    # Display Math Formulas
    st.subheader("Mathematical Analysis")
    col1, col2, col3 = st.columns(3)
    with col1:
        st.latex(f"f(x, y) = {sp.latex(f_expr)}")
    with col2:
        st.latex(r"\frac{\partial f}{\partial x} = " + sp.latex(fx_expr))
    with col3:
        st.latex(r"\frac{\partial f}{\partial y} = " + sp.latex(fy_expr))

    # --- 4. VISUALIZATION (Plotly) ---
    
    # Generate Grid Data
    x_vals = np.linspace(x_range[0], x_range[1], resolution)
    y_vals = np.linspace(y_range[0], y_range[1], resolution)
    X, Y = np.meshgrid(x_vals, y_vals)
    Z = f_func(X, Y)
    U = fx_func(X, Y) # Gradient X component
    V = fy_func(X, Y) # Gradient Y component

    tab1, tab2 = st.tabs(["3D Surface & Gradient", "2D Contour & Vector Field"])

    with tab1:
        st.write("### 3D View: Surface and Steepest Ascent")
        # 3D Surface Plot
        fig_3d = go.Figure(data=[go.Surface(z=Z, x=X, y=Y, colorscale='Viridis', opacity=0.8)])
        fig_3d.update_layout(title='Surface Plot of f(x,y)', autosize=True, height=600)
        st.plotly_chart(fig_3d, use_container_width=True)

    with tab2:
        st.write("### 2D View: Gradient Vector Field")
        # Quiver Plot (Vector Field) using Figure Factory
        fig_quiver = ff.create_quiver(x_vals, y_vals, U, V,
                                      scale=.1,
                                      arrow_scale=.3,
                                      name='Gradient Vectors',
                                      line_color='black')
        
        # Add Contour underneath
        fig_quiver.add_trace(go.Contour(
            z=Z, x=x_vals, y=y_vals,
            colorscale='Viridis',
            opacity=0.5,
            showscale=True
        ))
        
        fig_quiver.update_layout(
            title=f'Gradient Field over Contours (Arrows point uphill)',
            xaxis_title='x',
            yaxis_title='y',
            height=700,
            width=700
        )
        st.plotly_chart(fig_quiver, use_container_width=True)

except Exception as e:
    st.error(f"Error parsing function. Please check your syntax.\nDetails: {e}")

# --- 5. FOOTER & EXPLANATION ---
with st.expander("How this App uses AI & Python (For Report)"):
    st.write("""
    1. **Natural Input Parsing**: The app uses string manipulation to convert user-friendly inputs (like x³) into Python-compatible code (x**3).
    2. **Symbolic Math**: It uses the `SymPy` library to analytically calculate derivatives ($\partial f/\partial x, \partial f/\partial y$), ensuring mathematical precision.
    3. **Interactive Visualization**: `Plotly` is used to render the 3D surface and the 2D gradient field dynamically.
    """)
