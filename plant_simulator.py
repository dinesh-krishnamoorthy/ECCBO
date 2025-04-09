import numpy as np
from casadi import *


def EnterpriseA(par,Ts=1):
    '''
    - Benchmark Williams-Otto reactor model
    - ODE model
    - Williams, T.J. and Otto, R.E., 1960. A generalized chemical processing model for the investigation of computer control. Transactions of the American Institute of Electrical Engineers, Part I: Communication and Electronics, 79(5), pp.458-473.
    '''
    xa = MX.sym('xa')
    xb = MX.sym('xb')
    xc = MX.sym('xc')
    xp = MX.sym('xp')
    xe = MX.sym('xe')
    xg = MX.sym('xg')
    Tr = MX.sym('Tr')

    Fa = MX.sym('Fa')

    Fb = MX.sym('Fb')
    
    k01 = par.k01
    k02 = par.k02
    k03 = par.k03
    B1 = par.B1
    B2 = par.B2
    B3 = par.B3
    Vr = par.Vr
    Vh = par.Vh
    UA = par.UA
    cp = par.cp
    cp_h = par.cp_h
    rho_r = par.rho_r  # cold fluid density
    rho_h = par.rho_h   # hot fluid density

    
    
    k1 = k01*np.exp(-B1/(Tr))
    k2 = k02*np.exp(-B2/(Tr))
    k3 = k03*np.exp(-B3/(Tr))

    mr = rho_r*Vr
    mh = rho_h*Vh

    # PI loop to control xG = 0.08 
    CC_tau1 = 150
    CC_tauC = 60
    CC_Kp = CC_tau1/(0.00517*CC_tauC)
    CC_xG_sp = 0.08

    dxa = (Fa - (Fa+Fb)*xa - mr*xa*xb*k1)/mr
    dxb = (Fb - (Fa+Fb)*xb - mr*xa*xb*k1 - mr*xb*xc*k2)/mr
    dxc = -(Fa+Fb)*xc/mr + 2*xa*xb*k1 - 2*xb*xc*k2 - xc*xp*k3
    dxp = -(Fa+Fb)*xp/mr + xb*xc*k2 - 0.5*xp*xc*k3
    dxe = -(Fa+Fb)*xe/mr + 2*xb*xc*k2
    dxg = -(Fa+Fb)*xg/mr + 1.5*xp*xc*k3
    dTr = -CC_Kp*dxg + CC_Kp/CC_tau1*(CC_xG_sp-xg)
    
    diff = vertcat(dxa,dxb,dxc,dxp,dxe,dxg,dTr)
    x_var = vertcat(xa,xb,xc,xp,xe,xg,Tr)
    p_var = vertcat(Fb)
    d_var = vertcat(Fa)
    
    # objective
    L = -(1043.38*xp*(Fa+Fb)+20.92*xe*(Fa+Fb) - 118.34*Fb  -79.23*Fa)
    
    # Bounds
    lbu = [0.2]
    ubu = [5]
    lbx = [0,0,0,0,0,0,293]
    ubx = [0.12,1,1,1,1,0.08,500]
    lbw = vertcat(lbx,lbu)
    ubw = vertcat(ubx,ubu)

    # Create system
    sys = {'w':vertcat(x_var,p_var),'d':d_var,'f':diff,'L':L,
            'lbw':lbw,'ubw':ubw}

    # create CVODES integrator
    ode = {'x':x_var,'p':vertcat(p_var,d_var),'ode':diff,'quad':L}
    opts = {'tf':Ts}
    F = integrator('F','cvodes',ode,opts) 

    return sys,F

def EnterpriseB(par,Ts):
    '''
    - Williams-Otto reactor model with only 2 reactions (component C unmodelled)
    - ODE model
    - Zhang, Y. and Forbes, J.F., 2000. Extended design cost: a performance criterion for real-time optimization systems. Comput & Chem Eng, 24(8), pp.1829-1841.

    
    5 states - xa, xb, xp, xe, xg
    2 inputs - Fb, Tr
    1 disturbance - Fa
    2 parameters - k01, k02
    '''

    xa = MX.sym('xa')
    xb = MX.sym('xb')
    xp = MX.sym('xp')
    xe = MX.sym('xe')
    xg = MX.sym('xg')
    Tr = MX.sym('Tr')
    Th = MX.sym('Th')

    Fa = MX.sym('Fa')
    Tin = MX.sym('Tin')
    Thin = MX.sym('Thin')

    Fb = MX.sym('Fb')
    Fh = MX.sym('Fh')
    
    k01 = par.k01
    k02 = par.k02
    B1 = par.B1
    B2 = par.B2
    Vr = par.Vr
    Vh = par.Vh
    UA = par.UA
    cp = par.cp
    cp_h = par.cp_h
    rho_r = par.rho_r  # cold fluid density
    rho_h = par.rho_h   # hot fluid density

    T1 = Thin - Tr
    T2 = Th - Tin
    dTlm = (T1*T2*((T1+T2)/2))**(1/3)
    
    k1 = k01*np.exp(-B1/(Tr))
    k2 = k02*np.exp(-B2/(Tr))

    mr = rho_r*Vr
    mh = rho_h*Vh

    dxa = (Fa - (Fa+Fb)*xa - mr*xa*xb**2*k1 - mr*xa*xb*xp*k2)/mr
    dxb = (Fb - (Fa+Fb)*xb - 2*mr*xa*xb**2*k1 - mr*xa*xb*xp*k2)/mr
    dxp = -(Fa+Fb)*xp/mr + xa*xb**2*k1 - xa*xb*xp*k2
    dxe = -(Fa+Fb)*xe/mr + 2*xa*xb**2*k2
    dxg = -(Fa+Fb)*xg/mr + 3*xa*xb*xp*k2
    dTr = (Fa+Fb)*(Tin-Tr)/mr  + UA*dTlm/(mr*cp)
    dTh = Fh*(Thin-Th)/mh - UA*dTlm/(mh*cp_h)
    
    diff = vertcat(dxa,dxb,dxp,dxe,dxg,dTr,dTh)
    x_var = vertcat(xa,xb,xp,xe,xg,Tr,Th)
    p_var = vertcat(Fa,Fb,Fh)
    d_var = vertcat(Tin,Thin)
    
    # objective
    L = -(1043.38*xp*(Fa+Fb)+20.92*xe*(Fa+Fb) - 79.23*Fa - 118.34*Fb - 0*Fh - (Fa+Fb-5))
    
    # NL constraints
    g1 = xa - 0.12
    g2 = xg - 0.08
    g = vertcat(g1,g2)

    # Bounds
    lbu = [0,0,0]
    ubu = [102,10,10]
    lbx = [0,0,0,0,0,273,273]
    ubx = [1,1,1,1,1,500,500]
    lbw = vertcat(lbx,lbu)
    ubw = vertcat(ubx,ubu)

    # coupling constraint
    c = Function('c',[vertcat(x_var,p_var),d_var],[Fh],['w','d'],['c'])

    # Create system
    sys = {'w':vertcat(x_var,p_var),'d':d_var,'f':diff,'L':L,'g':g,'c':c,
            'lbw':lbw,'ubw':ubw}

    # create CVODES integrator
    ode = {'x':x_var,'p':vertcat(p_var,d_var),'ode':diff,'quad':L}
    opts = {'tf':Ts}
    F = integrator('F','cvodes',ode,opts) 
    return sys,F
   
def HeatingStation():
    Fh = MX.sym('Fh')
    Fmax = MX.sym('Fmax') 

    # objective
    L = -0.01*Fh

    #nlcon 
    g = -Fh-Fmax

    # coupling constraint
    c = Function('c',[Fh,Fmax],[Fh],['w','d'],['c'])

    # Create system
    sys = {'w':Fh,'d':Fmax,'f':vertcat([]),'L':L,'g':g,'c':c,
            'lbw':-100,'ubw':0}
    return sys

def ExothermicCSTR(par, Ts = 1):

    tau = par.tau       # Residence time (min)
    A1 = par.A1         # Pre-exponential factor for reaction 1 (1/s)
    A2 = par.A2         # Pre-exponential factor for reaction 2 (1/s)
    E1 = par.E1    # Activation energy for reaction 1 (cal/mol)
    E2 = par.E2      # Activation energy for reaction 2 (cal/mol)
    R = par.R        # Universal gas constant (cal/mol*K)
    Hr = par.Hr # Heat of reaction (cal/mol)
    rho = par.rho     # Density of the reaction mixture (kg/L)
    cp = par.cp

     # Define state variables: CA, CB, T
    CA = MX.sym('CA')
    CB = MX.sym('CB')
    T = MX.sym('T')
    Tin = MX.sym('Tin')

    CAin = MX.sym('CAin')
    CBin = MX.sym('CBin')
    Tsp = MX.sym('Tsp')
    F = MX.sym('F')

    TC_Kp = 0.0167
    TC_tau1 = 1

    # Define the reaction rates
    rate1 = A1 * np.exp(-E1 / (R * T)) * CA
    rate2 = A2 * np.exp(-E2 / (R * T)) * CB

    # Define the ODEs
    dCA = F * (CAin - CA) / tau - (rate1 - rate2)
    dCB = F * (CBin - CB) / tau + (rate1 - rate2)
    dT = F * (Tin - T) / tau + Hr * (rate1 - rate2) / (rho * cp)
    dTin = -TC_Kp*dT + TC_Kp/TC_tau1*(Tsp-T)

    diff = vertcat(dCA,dCB,dT,dTin)
    x_var = vertcat(CA,CB,T,Tin)
    p_var = vertcat(Tsp,F)
    d_var = vertcat(CAin,CBin)

    # objective
    L = -F -2.009*CB + (1.657e-3*Tin)**2 

    # Create system
    sys = {'w':vertcat(x_var,p_var),'d':d_var,'f':diff,'L':L}

    # create CVODES integrator
    ode = {'x':x_var,'p':vertcat(p_var,d_var),'ode':diff,'quad':L}
    opts = {'tf':Ts}
    F = integrator('F','cvodes',ode,opts) 
    return sys,F


class CSTR_parameters():
    tau = 60       # Residence time (min)
    A1 = 5000         # Pre-exponential factor for reaction 1 (1/s)
    A2 = 1e6         # Pre-exponential factor for reaction 2 (1/s)
    E1 = 10000.0      # Activation energy for reaction 1 (cal/mol)
    E2 = 15000.0      # Activation energy for reaction 2 (cal/mol)
    R = 1.987        # Universal gas constant (cal/mol*K)
    Hr = 5000.0    # Heat of reaction (cal/mol)
    rho = 1.0     # Density of the reaction mixture (kg/L)
    cp = 1000        # Heat capacity (J/g*K)

class EnterpriseA_parameters():
    k01 = 1.6599e6
    k02 = 7.2117e8
    k03 = 2.6745e12
    B1 = 6666.7
    B2 = 8333.3
    B3 = 11111
    Vr = 2.63
    Vh = 0.6
    UA = 2500
    cp = 1000
    cp_h = 4200
    rho_r = 800
    rho_h = 890


class EnterpriseB_parameters():
    k01 = 1.655e8
    k02 = 2.611e13
    B1 = 8077.6 # degK
    B2 = 12438.5
    UA = 2500
    cp = 1000
    cp_h = 4200
    Vr = 2.2
    Vh = 0.6
    rho_r = 950
    rho_h = 890