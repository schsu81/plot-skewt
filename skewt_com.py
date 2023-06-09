#!/usr/bin/env python3
import numpy as np
#%%---------------------------------------------------------------------
def com_vp2mr(p,vp):
  return 0.622*vp/(p-vp)
#%%---------------------------------------------------------------------
def com_vp2sh(p,vp):
  return 0.622*vp/(p-(1.-0.622)*vp)
#%%---------------------------------------------------------------------
      
def satured_vapor(T):
  # T in C
  # es in hPa
  return 6.1078*np.exp(17.27*T/(T+237.3))
#%%---------------------------------------------------------------------
      
def com_theta(T,P):
  return (T+273.15)*(1000./P)**0.286 -273.15
#%%---------------------------------------------------------------------
def adiabatic_dry(Ts,Ps,invars,inname='P'):
  if np.iterable(invars):
    vs = invars
  else:
    vs = [invars]
  out = np.zeros(len(vs),dtype=float)
  match inname:
    case 'P': # out T
      out = (Ts+273.15)*(vs/Ps)**(287.104/1004.86) -273.15
    case 'T': # out P
      out = Ps*((vs+273.15)/(Ts+273.15))**(1004.86/287.104)
    case _:
      raise('Unknown inname')
  return out   
#%%---------------------------------------------------------------------
def com_thetae(ta,pa):
  theta = com_theta(ta,pa)
  es = satured_vapor(ta)
  mr = com_vp2mr(pa,es)
  return (theta+273.15)*np.exp(2675.*mr/(ta+273.15))  
#%%---------------------------------------------------------------------
def com_ta(thetae,p):
  if np.iterable(p):
    ps = p
  else:
    ps = [p]
  thetae0  = np.array(thetae+273.15,dtype=float)
  
  ta = np.zeros(len(ps),dtype=float)
  for i in range(len(ps)):
    if ps[i]<100.:
      ta[i] = thetae*(ps[i]/1000.)
      continue
    loops = 0
    det   = -10.
    diff  = 1.
    ldiff = 1.
    while abs(diff)>0.001 and loops<5000:
      ta[i]    = ta[i]+det
      thetae_a = com_thetae(ta[i],ps[i])
      diff = thetae_a-thetae0
      if diff*ldiff<0:
        det /= -10.
      ldiff = diff    
      loops +=1  
  return ta
#%%=====================================================================
def adiabatic_psudo(Ts,Ps,invars,inname='P'):
  if np.iterable(invars):
    vs = invars
  else:
    vs = [invars]    
  thetae0  = com_thetae(Ts,Ps)
  match inname:
    case 'P': # out T
      out = np.zeros(len(vs),dtype=float)
      for i in range(len(vs)):
        if vs[i]<100.:
          out[i] = thetae0*(vs[i]/1000.)
          continue
        loops = 0
        det   = -10.
        diff  = 1.
        ldiff = com_thetae(out[i],vs[i])-thetae0
        while abs(diff)>0.001 and loops<5000:
          out[i]    = out[i]+det
          thetae_a = com_thetae(out[i],vs[i])
          diff = thetae_a-thetae0
          if abs(diff)>abs(ldiff):
            det /= -10.
          ldiff = diff    
          loops +=1  
    case 'T': # out P
      out = np.full(len(vs),Ps,dtype=float)
      for i in range(len(vs)):
        loops = 0
        det   = -100.
        diff  = 1.
        ldiff = com_thetae(vs[i],out[i])-thetae0
        while abs(diff)>0.001 and loops<5000:
          out[i]   = out[i]+det          
          thetae_a = com_thetae(vs[i],out[i])
          diff = thetae_a-thetae0
          if abs(diff)>abs(ldiff):
            det /= -10.
          ldiff = diff    
          loops +=1  
    case _:
      raise('Unknown type')
  return out      
adiabatic_moist = adiabatic_psudo
#%%---------------------------------------------------------------------
def saturated_ratio(ws,var,inname='P'):
  # T in C
  # P in hPa
  # ws in kg/kg
  match inname:
    case 'P':
      #out = np.log(ws*var/6.1078/0.622)
      out = np.log(ws*var/6.1078/(ws+0.622))
      out = (237.3*out)/(17.27-out)   
    case 'T':
      print(ws)
      #out = 6.1078*(ws+0.622)/ws*np.exp(17.27*var/(var+237.3))
      out = (ws+0.622)/ws*6.1078*np.exp(17.27*var/(237.3+var))
    case _:
      raise('Unknown type')
  return out
#%%=====================================================================
