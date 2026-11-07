




import streamlit as st
import numpy as np
import sympy as sp
from scipy.integrate import odeint
import matplotlib.pyplot as plt

# Configuración de la estructura de la página web
st.set_page_config(page_title="Simulador Sistemas 2x2", layout="wide")

st.title("📊 Simulador Científico de Sistemas 2x2")
st.markdown("Introduce las ecuaciones simbólicas, parámetros y condiciones iniciales para simular y analizar la estabilidad.")

# Diseño de dos columnas en la interfaz para organizar el formulario
col1, col2 = st.columns(2)

with col1:
    st.header("1. Ecuaciones Simbólicas")
    st.caption("Usa sintaxis matemática de Python (ej: b0*y - b1*y*z). Las variables obligatorias son 'y' y 'z'.")
    eq_dydt_str = st.text_input("dy/dt =", value="b0*y - b1*y*z")
    eq_dzdt_str = st.text_input("dz/dt =", value="-a0*z + a1*y*z")

with col2:
    st.header("2. Parámetros y Condiciones")
    params_str = st.text_input("Valores de los parámetros (Diccionario de Python):", 
                               value="{'b0': 1.1, 'b1': 0.4, 'a1': 0.1, 'a0': 0.4}")
    
    c_init1, c_init2 = st.columns(2)
    with c_init1:
        y0 = st.number_input("y0 inicial:", value=10.0, step=0.1)
    with c_init2:
        z0 = st.number_input("z0 inicial:", value=1.0, step=0.1)
        
    c_t, c_eval = st.columns(2)
    with c_t:
        t_max = st.number_input("Tiempo Máximo (t):", value=50, step=1)
    with c_eval:
        eval_pt_str = st.text_input("Punto de Evaluación [y, z]:", value="[4.0, 2.75]")

# Botón disparador para ejecutar los cálculos en el servidor de Streamlit
if st.button("🚀 Calcular y Graficar", type="primary"):
    try:
        # Parsear las entradas textuales del usuario a tipos nativos de Python
        params = eval(params_str)
        pt_eval = eval(eval_pt_str)
        
        # --- ANÁLISIS SIMBÓLICO CON SYMPY ---
        y_sym, z_sym = sp.symbols('y z')
        param_symbols = {name: sp.symbols(name) for name in params.keys()}
        local_dict = {'y': y_sym, 'z': z_sym, **param_symbols}
        
        # Parsear las fórmulas dinámicas
        f_sym = sp.parse_expr(eq_dydt_str, local_dict=local_dict)
        g_sym = sp.parse_expr(eq_dzdt_str, local_dict=local_dict)
        
        # Inyectar los valores fijos de los parámetros numéricos
        f_num_param = f_sym.subs(params)
        g_num_param = g_sym.subs(params)
        
        # Matriz Jacobiana, Traza y Determinante algebraicos
        Jacobiano = sp.Matrix([f_num_param, g_num_param]).jacobian([y_sym, z_sym])
        traza_simbolica = Jacobiano.trace()
        det_simbolico = Jacobiano.det()
        
        # Calcular los Estados Estacionarios (Puntos de Equilibrio)
        puntos_equilibrio = sp.solve([f_num_param, g_num_param], (y_sym, z_sym))
        
        # Extraer coordenadas del punto de evaluación seleccionado
        y_eval, z_eval = float(pt_eval[0]), float(pt_eval[1])
        J_evaluado = Jacobiano.subs({y_sym: y_eval, z_sym: z_eval})
        traza_num = traza_simbolica.subs({y_sym: y_eval, z_sym: z_eval})
        det_num = det_simbolico.subs({y_sym: y_eval, z_sym: z_eval})
        
        # --- DESPLIEGUE DE RESULTADOS MATEMÁTICOS ---
        st.header("📋 Resultados del Análisis")
        st.metric(label="Estados Estacionarios Encontrados (y, z)", value=str(puntos_equilibrio))
        
        res_col1, res_col2 = st.columns(2)
        with res_col1:
            st.subheader(f"Análisis en el punto crítico ({y_eval}, {z_eval})")
            st.text(f"Matriz Jacobiana evaluada:\n{np.array(J_evaluado).astype(float)}")
        with res_col2:
            st.metric(label="Traza (Tr)", value=f"{float(traza_num):.4f}")
            st.metric(label="Determinante (Det)", value=f"{float(det_num):.4f}")
            
        # --- RESOLUCIÓN NUMÉRICA CON SCIPY ---
        # Convertir ecuaciones de Sympy a lambdas numéricas ultrarrápidas
        f_lambda = sp.lambdify((y_sym, z_sym), f_num_param, 'numpy')
        g_lambda = sp.lambdify((y_sym, z_sym), g_num_param, 'numpy')
        
        def sistema_dinamico(x_vec, t_val):
            cur_y, cur_z = x_vec[0], x_vec[1]
            return [f_lambda(cur_y, cur_z), g_lambda(cur_y, cur_z)]
        
        t_array = np.linspace(0, t_max, 1000)
        solucion = odeint(sistema_dinamico, [y0, z0], t_array)
        y_sol = solucion[:, 0]
        z_sol = solucion[:, 1]
        
        # --- GENERACIÓN DE GRÁFICOS (MATPLOTLIB) ---
        st.header("📈 Gráficos del Sistema")
        g_col1, g_col2 = st.columns(2)
        
        with g_col1:
            fig1, ax1 = plt.subplots(figsize=(6, 4.5))
            ax1.plot(t_array, y_sol, color='k', linestyle='--', label='Variable y')
            ax1.plot(t_array, z_sol, color='b', label='Variable z')
            ax1.set_xlabel('t')
            ax1.set_ylabel('Valores de y, z')
            ax1.set_title('Evolución Temporal')
            ax1.legend()
            ax1.grid(True, linestyle=':')
            st.pyplot(fig1)
            
        with g_col2:
            fig2, ax2 = plt.subplots(figsize=(6, 4.5))
            ax2.plot(y_sol, z_sol, color='g')
            ax2.set_xlabel('Variable y')
            ax2.set_ylabel('Variable z')
            ax2.set_title('Plano de Fase: y vs z')
            ax2.grid(True, linestyle=':')
            st.pyplot(fig2)
            
    except Exception as e:
        st.error(f"❌ Ocurrió un error en la simulación. Verifica las expresiones matemáticas o parámetros. Detalle: {e}")
