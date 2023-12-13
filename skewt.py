import numpy as np
import warnings
float=np.float32
g       = 9.81
Rd      = 287.104
Cp      = 1004.86
epsilon = 0.622
#============================================================================
class soundingdata(dict):
  def __init__(self,P,T,Td,H=None,WS=None,WD=None,U=None,V=None,N=1000):
    self['P' ] = np.array(P ,dtype=float)
    self['T' ] = np.array(T ,dtype=float)
    self['Td'] = np.array(Td,dtype=float)
    self['H' ] = np.array(H ,dtype=float)
    self['WS'] = np.array(WS,dtype=float)
    self['WD'] = np.array(WD,dtype=float)
    self['U' ] = np.array(U ,dtype=float)
    self['V' ] = np.array(V ,dtype=float)
    self['hasH'] = H is not None 
    if (U is None) and (WS is not None ):
      self['U' ],self['V' ] = wd2uv(WS,WD)
    if (U is not None) and (WS is None ):
      self['WS'],self['WD'] = uv2wd(U,V)
    self['hasW'] = not np.isnan(self['WS']).all()

    for i in range(len(T)):
      if not np.isnan(T[i]):
        break

    self['parcel'] = self.get_parcel(P[i],N=N)
    for key in ['state','LCL','CCL','LFC','EL','CIN','CAPE']:
      self[key] = self['parcel'][key]
    self._com_indexes()
    self._com_qpf()
  #--------------------------------------------------------------------------
  def _com_indexes(self):
    P  = self['P']
    T  = self['T']
    Td = self['Td']
    N  = len(P)
    hasW = self['hasW']
    
    WS850,WD850,WS500,WD500=0.,0.,0.,0.

    i = 0
    if 850. in P:
      i, = np.where(P==850.)[0]
      T850 , Td850  = T[i],Td[i]
      if hasW:
        WS850, WD850  = self['WS'][i]*1.94, self['WD'][i]
    else:
      for i in range(i,N-1):
        if (P[i]>=850.) and (P[i+1]<850.):
          T850  = inter_logpT(P[i],T [i],P[i+1],T [i+1],850.)
          Td850 = inter_logpT(P[i],Td[i],P[i+1],Td[i+1],850.)
          break

    if 700. in P:
      i, = np.where(P==700.)[0]
      T700, Td700  = T[i],Td[i]
    else:
      for i in range(i,N-1):
        if (P[i]>=700.) and (P[i+1]<700.):
          T700  = inter_logpT(P[i],T [i],P[i+1],T [i+1],700.)
          Td700 = inter_logpT(P[i],Td[i],P[i+1],Td[i+1],700.)
          break

    if 500. in P:
      i, = np.where(P==500.)[0]
      T500, Td500  = T[i],Td[i]
      if hasW:
        WS500, WD500  = self['WS'][i]*1.94, self['WD'][i]
    else:
      for i in range(i,N-1):
        if (P[i]>=500.) and (P[i+1]<500.):
          T500  = inter_logpT(P[i],T [i],P[i+1],T [i+1],500.)
          Td500 = inter_logpT(P[i],Td[i],P[i+1],Td[i+1],500.)
          break

    self['KI']  = T850-T500+Td850-(T700-Td700)
    self['TTI'] = T850-T500+Td850-T500

    Tlcl = self['LCL']['T']
    Plcl = self['LCL']['P']
    Ta = adiabatic_pseudo(Plcl,Tlcl,500.,inname='P',guess=T500)
    self['LI'] = T500-Ta

    Tlcl = com_Tic(T850,Td850)
    Plcl = adiabatic_dry(850.,T850,Tlcl,inname='T') 
    Ta = adiabatic_pseudo(Plcl,Tlcl,500.,inname='P',guess=T500)
    self['SI'] = T500-Ta
    del Tlcl,Plcl

    if hasW:
      SWEAT = 2*WS850+WS500
      if Td850>0:
        SWEAT += 12*Td850
      if self['TTI']>49.:
        SWEAT += 20*(self['TTI']-49.)
      if     (WD850>=130.) and (WD850<=250.) and (WD500>=210.) and (WD500<=310.) \
         and (WS850>=15.  ) and (WS500>=15.):
        SWEAT += 125*(np.sin(np.deg2rad(WD500-WD850))+0.2)
    else:
      warnings.warn('No wind fields, skip computing SWEAT.')
      SWEAT = np.nan
    self['SWEAT'] = SWEAT
  #--------------------------------------------------------------------------
  def _com_qpf(self):
    i = np.sum(self['P']>300.)+1
    P  = self['P' ][:i]
    Td = self['Td'][:i]
    self['QPF'] = np.trapz(-satured_sh(P,Td)/g,P*100.)
  #--------------------------------------------------------------------------
  def get_parcel(self,Pstart,Pend=100.,N=1000,inverse=False):
    return parcel(self['P'],self['T'],self['Td'],self['H'],Pstart,Pend,N,inverse)
#============================================================================
class energy(dict):
  def __init__(self,P=np.nan,Ta=np.nan,Te=np.nan,H=np.nan):
    self['P' ] = P
    self['Te'] = Te
    self['Ta'] = Ta
    self['H']  = H

    if np.isnan(P).all():
      self['value'] = 0.
    elif np.isnan(H).all():
      self['value'] = np.trapz(-Rd*np.abs(Ta-Te)            ,np.log(P))
    else:
      self['value'] = np.trapz(  g*np.abs(Ta-Te)/(Te+273.15),       H )
  #--------------------------------------------------------------------------
  def __call__(self):
    return self['value']
#============================================================================
class level(dict):
  def __init__(self,P,T,H):
    self['P'] = float(P)
    self['T'] = float(T)
    self['H'] = float(H)
#============================================================================
class parcel(dict):
  def __init__(self,P,T,Td,H,Pstart,Pend,N,inverse):
    self['state'] = level(np.nan,np.nan,np.nan)
    self['LCL' ] = level(np.nan,np.nan,np.nan)
    self['CCL' ] = level(np.nan,np.nan,np.nan)
    self['LFC' ] = level(np.nan,np.nan,np.nan)
    self['EL'  ] = level(np.nan,np.nan,np.nan)
    self['CAPE'] = energy()
    self['CIN']  = energy()
    self['hasH'] = not np.isnan(H).all()
    self['hasLCL'] = False
    self['hasCCL'] = False
    self['hasLFC'] = False
    self['hasEL' ] = False

    NN = len(P)
    P  = np.array(P ,dtype=float)
    T  = np.array(T ,dtype=float)
    Td = np.array(Td,dtype=float)
    if self['hasH']:
      H = np.array(H,dtype=float)

    if Pstart in P:
      i0, = np.where(Pstart==P)[0]
      P_s    = P [i0] # start P
      T_s_c  = T [i0] # start T in c
      Td_s_c = Td[i0] # start Td in c
      if self['hasH']:
        H_s = H[i0]  # start H
    else:
      i0 = sum((P-Pstart)>0.)
      P_s    = Pstart
      T_s_c  = inter_logpT(P[i0-1],T [i0-1],P[i0],T [i0],P_s)
      Td_s_c = inter_logpT(P[i0-1],Td[i0-1],P[i0],Td[i0],P_s)
      if self['hasH']:
        H_s = hyd(H[i0-1],T[i0-1],H[i0],T[i0],P_s,inname='P')

    if inverse:
      # find lcl and bottom
      ib = 0
      Tlcl = com_Tic(T[ib],Td[ib])
      Plcl   = adiabatic_dry   (P[ib],T[ib],Tlcl,inname='T')
      Tlcls  = adiabatic_pseudo(P_s  ,T_s_c,Plcl,inname='P',guess=Tlcl)
      dT     = Tlcls-Tlcl
      if dT == 0.:
        P_b    = P [ib]
        T_b_c  = T [ib]
        Td_b_c = Td[ib]
        self['hasLCL'] = True
      else:
        for ib in range(1,NN):
          ldT    = dT
          Tlcl   = com_Tic(T[ib],Td[ib])
          Plcl   = adiabatic_dry   (P[ib],T[ib],Tlcl,inname='T')
          Tlcls  = adiabatic_pseudo(P_s  ,T_s_c,Plcl,inname='P',guess=Tlcl)
          dT     = Tlcls-Tlcl
          if ldT*dT<=0.:
            self['hasLCL'] = True
            ib -= 1
            break
          if P[ib]<P_s:
            ib = 0
            P_b = P[0]
            T_b_c = adiabatic_pseudo(P_s,T_s_c,P_b,inname='P',guess=T[0])
            break
        if self['hasLCL']:
          step   = (P[ib+1]-P[ib])/10.
          P_b    = P[ib]
          T_b_c  = inter_logpT(P[ib],T [ib],P[ib+1],T [ib+1],P_b)
          Td_b_c = inter_logpT(P[ib],Td[ib],P[ib+1],Td[ib+1],P_b)
          Tlcl   = com_Tic(T_b_c,Td_b_c)
          Plcl   = adiabatic_dry   (P_b,T_b_c,Tlcl,inname='T')
          Tlcls  = adiabatic_pseudo(P_s,T_s_c,Plcl,inname='P',guess=Tlcl)
          dT     = np.abs(Tlcls-Tlcl)
          t   = 0
          while (dT>=0.001) and (t<1000) :
            ldT = dT
            P_b   += step
            T_b_c  = inter_logpT(P[ib],T [ib],P[ib+1],T [ib+1],P_b)
            Td_b_c = inter_logpT(P[ib],Td[ib],P[ib+1],Td[ib+1],P_b)
            Tlcl   = com_Tic(T_b_c,Td_b_c)
            Plcl   = adiabatic_dry   (P_b,T_b_c,Tlcl,inname='T')
            Tlcls  = adiabatic_pseudo(P_s,T_s_c,Plcl,inname='P',guess=Tlcl)
            dT = np.abs(Tlcls-Tlcl)
            if dT>ldT:
              step = -step/10.
            t  +=1
    else:
      ib     = i0
      P_b    = P_s
      T_b_c  = T_s_c
      Td_b_c = Td_s_c
      # LCL
      self['hasLCL'] = True
      Tlcl = com_Tic(T_s_c,Td_s_c)
      Plcl = adiabatic_dry(P_s,T_s_c,Tlcl,inname='T') 

    # CCL
    if self['hasLCL']:
      self['hasCCL'] = True
      mr  = satured_mr(P_b,Td_b_c)
      mr0 = satured_mr(P_b,T_b_c)
      for i in range(ib+1,NN):
        mr1 = satured_mr(P[i],T[i])
        if (mr<=mr0) and (mr>mr1):
          deta = (Plcl-P_b)*(T[i]-T[i-1]) - (P[i]-P[i-1])*(Tlcl-Td_b_c)
          if abs(deta)>0:
            deta = ( (P[i-1]-P_b)*(Tlcl-Td_b_c)-(Plcl-P_b)*(T[i-1]-Td_b_c) )/deta
            Tccl = T[i-1]+(T[i]-T[i-1])*deta 
            Pccl = P[i-1]+(P[i]-P[i-1])*deta
            self['CCL']['T'] = float(Tccl)
            self['CCL']['P'] = float(Pccl)
            if self['hasH']:
              Hccl = inter_hydH(P[i-1:i+1],T[i-1:i+1],H[i-1:i+1],Pccl,Tccl)
              self['CCL']['H'] = float(Hccl)
          del deta
          break
        mr0 = mr1
      del mr,mr0,mr1

    # compute profile
    if inverse:
      TP = P_s
    else:
      TP = max(P[-1],Pend)
    P_a   = np.logspace(np.log10(P_b),np.log10(TP),N,dtype=float) # P profile
    T_e_c = np.full(P_a.shape[0],np.nan,dtype=float) # T environment profile in c
    T_a_c = np.full(P_a.shape[0],np.nan,dtype=float) # T air parcel profile in c
    H_a   = np.full(P_a.shape[0],np.nan,dtype=float) # H profile
    P_a[0],T_e_c[0],T_a_c[0] = P_b,T_b_c,T_b_c
    self['state']['P'] = P_a  [0]
    self['state']['T'] = T_a_c[0]
    if self['hasH']:
      H_a[0] = inter_hydH(P[ib:ib+2],T[ib:ib+2],H[ib:ib+2],P_b,T_b_c) 
      self['state']['H'] = H_a  [0]

    j = 0
    for i in range(1,P_a.shape[0]):
      while j<NN-1:
        if (P_a[i]<P[j]) and (P_a[i]>=P[j+1]):
          T_e_c[i] = inter_logpT(P[j],T[j],P[j+1],T[j+1],P_a[i])
          if self['hasLCL']:
            if P_a[i]>=Plcl:
              T_a_c[i] = adiabatic_dry(Plcl,Tlcl,P_a[i],inname='P')
            else:
              T_a_c[i] = adiabatic_pseudo(Plcl,Tlcl,P_a[i],inname='P',guess=T_a_c[i-1])
          else: # happen in inverse
            T_a_c[i] = adiabatic_pseudo(P_s,T_s_c,P_a[i],inname='P',guess=T_a_c[i-1])
          if self['hasH']:
            H_a[i] = inter_hydH(P[j:j+2],T[j:j+2],H[j:j+2],P_a[i],T_a_c[i])
          break
        else:
          j+=1

    # insert LCL
    if self['hasLCL']:
      i = np.where(P_a-Plcl<0.)[0][0]
      ilcl = i.copy()
      if self['hasH']:
        Hlcl = inter_hydH(P_a[i-1:i+1],T_e_c[i-1:i+1],H_a[i-1:i+1],Plcl,Tlcl)
        H_a   = np.insert(H_a,ilcl,Hlcl)
        self['LCL']['H'] = H_a[ilcl]
      T_e_c = np.insert(T_e_c,ilcl,inter_logpT(P_a[i-1],T_e_c[i-1],P_a[i],T_e_c[i],Plcl))
      P_a   = np.insert(P_a  ,ilcl,Plcl)
      T_a_c = np.insert(T_a_c,ilcl,Tlcl)
      self['LCL']['P'] = P_a  [ilcl]
      self['LCL']['T'] = T_a_c[ilcl]

      # insert LFC
      i, = np.where(T_a_c[ilcl+1:]-T_e_c[ilcl+1:]>=0.)
      if i.shape[0] != 0:
        self['hasLFC'] = True
        i    = i[0] + ilcl+1
        ilfc = i.copy()
        Plfc,Tlfc,Telfc = self._inter_level(P_a[i-1:i+1],T_a_c[i-1:i+1],T_e_c[i-1:i+1],Plcl,Tlcl)
        if self['hasH']:
          Hlfc = inter_hydH(P_a[i-1:i+1],T_e_c[i-1:i+1],H_a[i-1:i+1],Plfc,Tlfc)
          H_a   = np.insert(H_a,ilfc,Hlfc)
        T_e_c = np.insert(T_e_c,ilfc,Tlfc)
        P_a   = np.insert(P_a  ,ilfc,Plfc)
        T_a_c = np.insert(T_a_c,ilfc,Tlfc)
      elif inverse:
        self['hasLFC'] = True
        ilfc = P_a.shape[0]-1

      if self['hasLFC']:
        self['LFC']['P'] = P_a  [ilfc] 
        self['LFC']['T'] = T_a_c[ilfc]
        if self['hasH']:
          self['LFC']['H'] = H_a[ilfc]
        self['CIN']  = energy(P_a[:ilfc+1],T_a_c[:ilfc+1],T_e_c[:ilfc+1],H_a[:ilfc+1])

        # EL
        i, = np.where(T_a_c[ilfc+1:]-T_e_c[ilfc+1:]<=0.)
        if i.shape[0] != 0:
          self['hasEL'] = True
          i    = i[0] + ilfc+1
          iel = i.copy()
          Pel,Tel,Teel = self._inter_level(P_a[i-1:i+1],T_a_c[i-1:i+1],T_e_c[i-1:i+1],Plcl,Tlcl)
          if self['hasH']:
            Hel = inter_hydH(P_a[i-1:i+1],T_e_c[i-1:i+1],H_a[i-1:i+1],Pel,Tel)
            H_a   = np.insert(H_a,iel,Hel)
          T_e_c = np.insert(T_e_c,iel,Tel)
          P_a   = np.insert(P_a  ,iel,Pel)
          T_a_c = np.insert(T_a_c,iel,Tel)
        elif inverse and (ilfc!=P_a.shape[0]-1):
          self['hasEL'] = True
          iel = P_a.shape[0]-1
        if self['hasEL']:
          self['EL']['P'] = P_a  [iel] 
          self['EL']['T'] = T_a_c[iel] 
          if self['hasH']:
            self['EL']['H'] = H_a[iel]
          self['CAPE'] = energy(P_a[ilfc:iel +1],T_a_c[ilfc:iel +1],T_e_c[ilfc:iel +1],H_a[ilfc:iel +1])


    self['P'] = P_a
    self['T'] = T_a_c
  #--------------------------------------------------------------------------
  def __call__(self):
    return [self['T'],self['P']]
  #--------------------------------------------------------------------------
  def _inter_level(self,Pa,Ta,Te,Plcl,Tlcl):
    deta = (Ta[1]-Ta[0])-(Te[1]-Te[0])
    if deta==0.:
      return [np.nan]*3
    deta = (Te[0]-Ta[0])/deta
    Pa0  = np.exp( np.log(Pa[0]) + deta*np.log(Pa[1]/Pa[0]) )
    Te0  =                Ta[0]  + deta*(      Ta[1]-Ta[0])
    Ta0  = adiabatic_pseudo(Plcl,Tlcl,Pa0,inname='P',guess=Ta[0])

    if deta==0.:
      return [np.nan]*3
    deta = (Te[0]-Ta[0])/( (Ta0-Ta[0])-(Te0-Te[0]) )
    Pa1  = np.exp( np.log(Pa[0]) + deta*(np.log(Pa0/Pa[0])) )
    Te1  =                Ta[0]  + deta*(       Ta0-Ta[0])
    Ta1  = adiabatic_pseudo(Plcl,Tlcl,Pa1,inname='P',guess=Ta0)
    #print(Te1,Ta1)
    return Pa1,Ta1,Te1
#============================================================================
def uv2wd(inu,inv):
  u  = inu
  v  = inv
  ws = np.sqrt(u**2+v**2)
  wd = 270.-np.rad2deg(np.arctan2(v,u))
  wd = np.mod(wd+360.,360.) # avoid negtive
  wd[ws==0.] = 0.
  return [ws,wd]
#%%--------------------------------------------------------------------
def wd2uv(inws,inwd):
  ws = inws
  wd = np.deg2rad(inwd)
  u  = -ws*np.sin(wd)
  v  = -ws*np.cos(wd)
  return [u,v]
#%%--------------------------------------------------------------------
def vp2mr(P,Pv):
  return epsilon*Pv/(P-Pv)
#%%---------------------------------------------------------------------
def vp2sh(P,Pv):
  return epsilon*Pv/(P-(1.-epsilon)*Pv)
#%%---------------------------------------------------------------------
def satured_vapor(T):
  # T   in C
  # Pvs in hPa
  return 6.112*np.exp(17.67*T/(T+243.5))
  #return 6.1078*np.exp(17.27*T/(T+237.3))
#%%---------------------------------------------------------------------
def satured_mr(P,T):
  # P   in hPa
  # T   in C
  # Pvs in hPa
  Pvs = satured_vapor(T)
  return vp2mr(P,Pvs) 
#%%---------------------------------------------------------------------
def satured_sh(P,T):
  # P   in hPa
  # T   in C
  # Pvs in hPa
  Pvs = satured_vapor(T)
  return vp2sh(P,Pvs) 
#%%---------------------------------------------------------------------
def theta(P,T):
  return (T+273.15)*(1000./P)**(Rd/Cp) -273.15
#%%---------------------------------------------------------------------
def hyd(z0,t0,z1,t1,inz,inname='P'):
  Tm = (t0+t1)/2.+273.15
  match inname:
    case 'P': # return Z
      return inz+Rd*Tm*np.log(z0/z1)/g
    case 'Z': # return P
      return inz*np.exp(-g/Rd/Tm*(z1-z0)) 
    case _:
      raise('Unknow variable name.')
#%%---------------------------------------------------------------------
def theta2thetaE(P,T):
  To  = theta(P,T)
  Pvs = satured_vapor(T)
  mr  = vp2mr(P,Pvs)
  return (To+273.15)*np.exp(2675.*mr/(T+273.15))  
#%%---------------------------------------------------------------------
def com_Tic(T,Td): # isentropic condensation temperature[C]
  Tk  = T +273.15
  Tdk = Td+273.15
  return 1./(1./(Tdk-56.)+np.log(Tk/Tdk)/800.) +56. -273.15
#%%---------------------------------------------------------------------
# current not used
#def com_cwa_integral(x,y):
#  # parabolic curved fitting.
#  def _rsum(x,y,p,q):
#    dx  = np.array([x[0]-x[1],x[0]-x[2],x[1]-x[2]])
#    dx  = np.where(abs(dx)<=1E-10,1E-10,dx)
#    a =  y[0]/(dx[0]*dx[1])
#    b = -y[1]/(dx[0]*dx[2])
#    c =  y[2]/(dx[1]*dx[2])
#    return   (a+b+c)*(q**3-p**3)/3. \
#           - 0.5*( (x[1]+x[2])*a+(x[0]+x[2])*b+(x[0]+x[1])*c )*(q**2-p**2) \
#           +     (  x[1]*x[2] *a+ x[0]*x[2] *b+ x[0]*x[1] *c )*(q-p)
#  value = 0.
#  N = len(x)
#  for i in range(N-1):
#    if i==0:
#      value  += _rsum(x[i  :i+3],y[i  :i+3],x[i],x[i+1])
#      continue
#    elif i==N-2:
#      value  += _rsum(x[i-1:i+2],y[i-1:i+2],x[i],x[i+1])
#      continue
#    value  += 0.5 *( _rsum(x[i  :i+3],y[i  :i+3],x[i],x[i+1])
#                    +_rsum(x[i-1:i+2],y[i-1:i+2],x[i],x[i+1]))
#  return value 
#%%---------------------------------------------------------------------
def inter_hydH(p,t,h,pl,tl):
  out  = hyd(p[0],t[0],pl,tl,h[0],'P')
  out += hyd(p[1],t[1],pl,tl,h[1],'P')
  return out/2.
#%%---------------------------------------------------------------------
def inter_logpT(p0,t0,p1,t1,p):
  return t0+(t1-t0)/np.log(p1/p0)*np.log(p/p0)
  #return t0+(t1-t0)/(p1-p0)*(p-p0)
#%%=====================================================================
def saturated_ratio(mr,var,inname='P'):
  # T in C
  # P in hPa
  # mr in kg/kg
  match inname:
    case 'P':
      #out = np.log(mr*var/6.1078/epsilon)
      out = np.log(mr*var/6.1078/(mr+epsilon))
      out = (237.3*out)/(17.27-out)   
    case 'T':
      #out = 6.1078*(mr+epsilon)/mr*np.exp(17.27*var/(var+237.3))
      out = (mr+epsilon)/mr*6.1078*np.exp(17.27*var/(237.3+var))
    case _:
      raise('Unknown type')
  return out
#%%---------------------------------------------------------------------
def adiabatic_dry(Ps,Ts,invars,inname='P'):
  vs = invars
  if hasattr(invars,'shape'):
    out = np.zeros(vs.shape,dtype=float)
  else:
    out = np.zeros(len(vs),dtype=float)
  match inname:
    case 'P': # out T
      out = (Ts+273.15)*(vs/Ps)**(Rd/Cp) -273.15
    case 'T': # out P
      out = Ps*((vs+273.15)/(Ts+273.15))**(Cp/Rd)
    case _:
      raise('Unknown inname')
  return out   
#%%---------------------------------------------------------------------
def adiabatic_pseudo(Ps,Ts,invars,inname='P',guess=None):
  if np.iterable(invars):
    vs = invars
  else:
    vs = [invars]    
  Tes = theta2thetaE(Ps,Ts)
  match inname:
    case 'P': # out T
      out = np.zeros(len(vs),dtype=float)
      if guess is not None:
        out[0] = guess
      for i in range(len(vs)):
        if vs[i]<100.:
          out[i] = Tes*(vs[i]/1000.)
          continue
        if i>=1:
          out[i] = out[i-1]
        loops = 0
        det   = -10.
        diff  = 1.
        ldiff = theta2thetaE(vs[i],out[i])-Tes
        while abs(diff)>0.01 and loops<5000:
          out[i]    = out[i]+det
          diff = theta2thetaE(vs[i],out[i])-Tes
          if abs(diff)>abs(ldiff):
            det /= -10.
          elif (diff==ldiff):
            break
          ldiff = diff    
          loops +=1  
    case 'T': # out P
      out = np.full(len(vs),Ps,dtype=float)
      if guess is not None:
        out[0] = guess
      for i in range(len(vs)):
        if i>=1:
          out[i] = out[i-1]
        loops = 0
        det   = -100.
        diff  = 1.
        ldiff = theta2thetaE(out[i],vs[i])-Tes
        while abs(diff)>0.01 and loops<5000:
          out[i]   = out[i]+det          
          diff = theta2thetaE(out[i],vs[i])-Tes
          if abs(diff)>abs(ldiff):
            det /= -10.
          ldiff = diff    
          loops +=1  
    case _:
      raise('Unknown type')
  if out.shape[0]==1:
    return out[0]
  else:
    return out
adiabatic_moist = adiabatic_pseudo
#%%=====================================================================
"""
===========================================================
SkewT-logP diagram: using transforms and custom projections
===========================================================

This serves as an intensive exercise of Matplotlib's transforms and custom
projection API. This example produces a so-called SkewT-logP diagram, which is
a common plot in meteorology for displaying vertical profiles of temperature.
As far as Matplotlib is concerned, the complexity comes from having X and Y
axes that are not orthogonal. This is handled by including a skew component to
the basic Axes transforms. Additional complexity comes in handling the fact
that the upper and lower X-axes have different data ranges, which necessitates
a bunch of custom classes for ticks, spines, and axis to handle this.
"""

import matplotlib.pyplot as plt
from contextlib import ExitStack
from matplotlib.axes import Axes
from matplotlib.projections import register_projection
import matplotlib.transforms as transforms
import matplotlib.axis as maxis
import matplotlib.axes as maxes
import matplotlib.spines as mspines
from matplotlib.ticker import (MultipleLocator, NullFormatter,ScalarFormatter)

# The sole purpose of this class is to look at the upper, lower, or total
# interval as appropriate and see what parts of the tick to draw, if any.
class SkewXTick(maxis.XTick):
    def draw(self, renderer):
        # When adding the callbacks with `stack.callback`, we fetch the current
        # visibility state of the artist with `get_visible`; the ExitStack will
        # restore these states (`set_visible`) at the end of the block (after
        # the draw).
        with ExitStack() as stack:
            for artist in [self.gridline, self.tick1line, self.tick2line,
                           self.label1, self.label2]:
                stack.callback(artist.set_visible, artist.get_visible())
            needs_lower = transforms.interval_contains(
                self.axes.lower_xlim, self.get_loc())
            needs_upper = transforms.interval_contains(
                self.axes.upper_xlim, self.get_loc())
            self.tick1line.set_visible(
                self.tick1line.get_visible() and needs_lower)
            self.label1.set_visible(
                self.label1.get_visible() and needs_lower)
            self.tick2line.set_visible(
                self.tick2line.get_visible() and needs_upper)
            self.label2.set_visible(
                self.label2.get_visible() and needs_upper)
            super().draw(renderer)
            
    def get_view_interval(self):
        return self.axes.xaxis.get_view_interval()

# This class exists to provide two separate sets of intervals to the tick,
# as well as create instances of the custom tick
class SkewXAxis(maxis.XAxis):
    def _get_tick(self, major):
        return SkewXTick(self.axes, None, major=major)

    def get_view_interval(self):
        return self.axes.upper_xlim[0], self.axes.lower_xlim[1]

    def fillon(self,**kwargs):
      x = self.axes.get_xticks()
      y = self.axes.get_ylim()
      i0, = np.where(x==0.)[0] 
      for i in range((i0+1)%2,len(x)-1,2):
        self.axes.fill_betweenx(y,x[i],x[i+1],**kwargs)

# This class exists to calculate the separate data range of the
# upper X-axis and draw the spine there. It also provides this range
# to the X-axis artist for ticking and gridlines
class SkewSpine(mspines.Spine):
    def _adjust_location(self):
        pts = self._path.vertices
        if self.spine_type == 'top':
            pts[:, 0] = self.axes.upper_xlim
        else:
            pts[:, 0] = self.axes.lower_xlim

class LogYAxis(maxis.YAxis):
  def _scale(self):
    return LogScale(self)

# This class handles registration of the skew-xaxes as a projection as well
# as setting up the appropriate transformations. It also overrides standard
# spines and axes instances as appropriate.
class SkewXAxes(Axes):
    # The projection must specify a name.  This will be used be the
    # user to select the projection, i.e. ``subplot(projection='skewt')``.
    name = 'skewt'

    def _init_axis(self):
        # Taken from Axes and modified to use our modified X-axis
        self.xaxis = SkewXAxis(self)
        self.spines.top.register_axis(self.xaxis)
        self.spines.bottom.register_axis(self.xaxis)
        #self.yaxis = LogYAxis(self)
        self.yaxis = maxis.YAxis(self)
        #self.yaxis._set_autoscale_on(False)
        #self.yaxis._set_scale('log')
        self.spines.left.register_axis(self.yaxis)
        self.spines.right.register_axis(self.yaxis)

    def _gen_axes_spines(self):
        spines = {'top': SkewSpine.linear_spine(self, 'top'),
                  'bottom': mspines.Spine.linear_spine(self, 'bottom'),
                  'left': mspines.Spine.linear_spine(self, 'left'),
                  'right': mspines.Spine.linear_spine(self, 'right')}
        return spines

    def _set_lim_and_transforms(self):
        """
        This is called once when the plot is created to set up all the
        transforms for the data, text and grids.
        """
        rot = 53

        # Get the standard transform setup from the Axes base class
        super()._set_lim_and_transforms()

        # Need to put the skew in the middle, after the scale and limits,
        # but before the transAxes. This way, the skew is done in Axes
        # coordinates thus performing the transform around the proper origin
        # We keep the pre-transAxes transform around for other users, like the
        # spines for finding bounds
        self.transDataToAxes = (
            self.transScale
            + self.transLimits
            + transforms.Affine2D().skew_deg(rot, 0)
        )
        # Create the full transform from Data to Pixels
        self.transData = self.transDataToAxes + self.transAxes

        # Blended transforms like this need to have the skewing applied using
        # both axes, in axes coords like before.
        self._xaxis_transform = (
            transforms.blended_transform_factory(
                self.transScale + self.transLimits,
                transforms.IdentityTransform())
            + transforms.Affine2D().skew_deg(rot, 0)
            + self.transAxes
        )

    
    @property
    def lower_xlim(self):
        return self.axes.viewLim.intervalx

    @property
    def upper_xlim(self):
        pts = [[0., 1.], [1., 1.]]
        return self.transDataToAxes.inverted().transform(pts)[:, 0]

    def set_default(self):
      self.xaxis.set_major_locator(MultipleLocator(5))
      self.set_xlim(-20, 40)
      self.set_xlabel('Temperature [$^\circ$C]')

      self.set_yscale('log')
      self.invert_yaxis()
      self.yaxis.set_major_formatter(ScalarFormatter())
      self.set_ylim(1100, 100)
      self.set_ylabel('Pressure [hPa]')
      self.set_yticks([100,150,200,250,300,400,500,600,700,850,1000])

    def set_hlabels(self,P=None,H=None):
      if self.data['hasH']:
        axh = self.twinx()
        axh.set_yscale('log')
        axh.invert_yaxis()
        axh.set_ylim(*self.get_ylim())
        axh.set_ylabel('Height [m]')

        if P is not None:
          if H is not None:
            axh.set_yticks(P,H)
          else:
            ticks = P
        else:
          ticks  = self.get_yticks()

        if ticks[1] > ticks[0]:
          ticks = ticks[::-1]

        pticks = []
        labels = []
        lP = self.data['P']
        lT = self.data['T']
        lH = self.data['H']
        j = 0
        for i in range(len(ticks)):
          if ticks[i] in lP:
            j, = np.where(ticks[i]==lP)[0]
            pticks.append(ticks[i])
            labels.append('%.0d' % lH[j])
          else:
            for j in range(j,len(lP)-1):
              if (lP[j]>=ticks[i]) and (lP[j+1]<ticks[i]):
                t = inter_logpT(lP[j],lT[j],lP[j+1],lT[j+1],ticks[i])
                labels.append('%.0d' % hyd(lP[j],lT[j],ticks[i],t,lH[j],inname='P') )
                pticks.append(ticks[i])
                break

        axh.set_yticks(pticks,labels)
      else:
        warnings.warn('No height data, skip set_hlabels.')
    
    def fillon_x(self,**kwargs):
      x = self.get_xticks()
      y = self.get_ylim()
      i0, = np.where(x==0.)[0] 
      for i in range((i0+1)%2,len(x)-1,2):
        self.fill_betweenx(y,x[i],x[i+1],**kwargs)

    def dry_adiabat(self,p=np.logspace(np.log10(1100),np.log10(100),101,dtype=float),
                         t=np.arange( -20,200, 10,dtype=float),
                         fmt='-',**kwargs):
      for t0 in t:
        x = adiabatic_dry(1000.,t0, p)
        self.plot(x,p,fmt,**kwargs)

    def pseudo_adiabat(self,p=np.logspace(np.log10(1100),np.log10(100),101,dtype=float),
                            t=np.arange( -20, 45,  5,dtype=float),
                            fmt='-',**kwargs):
      for t0 in t:
        x = adiabatic_pseudo(1000.,t0, p,inname='P',guess=t0)
        self.plot(x,p,fmt,**kwargs)
    moist_adiabat = pseudo_adiabat

    def saturated_ratio(self, p=np.logspace(np.log10(1100),np.log10(200),101,dtype=float),
                             qr=[0.7,1,1.5,2,3,4,5,7,10,15,20,25,30],
                            fmt=':',line={},text={}):
      if 'y' in text.keys():
        ty = text.pop('y')
      else:
        ty = 1010.
      for q in qr:
        x = saturated_ratio(q/1000.,p)
        self.plot(x,p,fmt,**line)
        self.text(x[0],ty,q,**text)

    def set_data(self,*args,**kwargs):
      self.data  = soundingdata(*args,**kwargs)
    set_sounding = set_data 
    set_profile  = set_data

    def plotdata(self,key,*args,**kwargs):
      if key in self.data:
        if isinstance(self.data[key],dict):
          if 'T' in self.data[key]:
            self.plot(self.data[key]['T'],self.data[key]['P'],*args,**kwargs)
          elif ('Ta' in self.data[key]) and ('Te' in self.data[key]):
            self.fill_betweenx(self.data[key]['P'],self.data[key]['Ta'],self.data[key]['Te'],*args,**kwargs)
        else:  
          self.plot(self.data[key],self.data['P'],*args,**kwargs)
      elif key in ['wind','H']:
        match key:
          case 'wind':
            self._plotwind(*args,**kwargs)
      else:
        raise('keyword error.')

    def _plotwind(self,*args,**kwargs):
      if self.data['hasW']:
        loc =  kwargs.pop('loc',37.)
        if hasattr(self,'axw'):
          axw = self.axw
        else:
          axw = self.inset_axes([0.,0.,1.,1.],sharey=self)
          axw.set_xlim(*self.get_xlim())
          axw.set_ylim(*self.get_ylim())
          axw.set_axis_off()
          self.axw = axw
        axw.barbs(np.full(len(self.data['P']),loc),self.data['P'],
                  self.data['U']*1.94,self.data['V']*1.94,*args,**kwargs)
      else:
        warnings.warn('No wind fields, skip plotting wind barb.')


#%%
# Now register the projection with matplotlib so the user can select it.
register_projection(SkewXAxes)
#############################################################################
#
# .. admonition:: References
#
#    The use of the following functions, methods, classes and modules is shown
#    in this example:
#
#    - `matplotlib.transforms`
#    - `matplotlib.spines`
#    - `matplotlib.spines.Spine`
#    - `matplotlib.spines.Spine.register_axis`
#    - `matplotlib.projections`
#    - `matplotlib.projections.register_projection`
#############################################################################
#def add_skewt:
#  def __init__(self,fig=None,windpanel=True):
#    pass
