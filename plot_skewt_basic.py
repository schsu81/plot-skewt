#!/usr/bin/env python3

if __name__ == '__main__':
    # Now make a simple example using the custom projection.
    from io import StringIO
    from matplotlib.ticker import (MultipleLocator, NullFormatter,
                                   ScalarFormatter)
    import matplotlib.pyplot as plt
    import numpy as np
    import skewt
    #import skewt_com as scom
     
    with open('data.txt','r') as f:
      data_txt = f.read()
      sound_data = StringIO(data_txt)
      stno, date, flag, P, H, T, Td, WD, WS, RH = np.loadtxt(sound_data, unpack=True)
      del data_txt,sound_data
      T  = np.where(T ==999.9,np.nan,T)
      Td = np.where(Td==999.9,np.nan,Td)
      WD = np.where(WD==999. ,np.nan,WD)
      WS = np.where(WS==999.9,np.nan,WS)
      RH = np.where(WS==999. ,np.nan,RH)

    # Create a new figure. The dimensions here give a good aspect ratio
    fig = plt.figure(figsize=(8,9))
    ax  = fig.add_axes(111,projection='skewt')
    ax.set_default()

    # An example of a slanted line at constant X
    ax.axvline(0, color='black')

    ax.xaxis.fillon(color='yellow')
    ax.grid(axis='y',linewidth=1.0,color='black')
    
    ax.dry_adiabat(color='blue',linewidth=1.0)
    ax.pseudo_adiabat(color='green',linewidth=1.0)
    #ax.moist_adiabat(color='green',linewidth=1.0) # work as well as ax.pseudo_adiabat.
    ax.saturated_ratio(text={'y':1010.,'ha':'center','va':'top'},line={'color':'black','linewidth':1.0})

    Tt = np.linspace(30.,-90.,13)
    Pt = np.logspace(np.log10(925.),np.log10(120.),13)
    for i in range(len(Tt)):
      ax.text(Tt[i],Pt[i],Tt[i])

    ax.set_yticks([100,150,200,250,300,400,500,600,700,850,1000])

    #- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 
    ax.set_data(P,T,Td,H,WS,WD)
    #ax.set_data(P,T,Td)
    ax.set_hlabels()

    for key in ['KI','TTI','LI','SI','SWEAT','QPF']:
      print(f'{key}=',ax.data[key])
    #- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 

    ax.plot(ax.data['T' ],ax.data['P'],color='blue',linewidth=2.,zorder=10)
    ax.plot(ax.data['Td'],ax.data['P'],color='red',linewidth=2.,zorder=10)

    # get axis for windbarb
    axw = ax.inset_axes([0.,0.,1.,1.],sharey=ax)
    axw.set_xlim(*ax.get_xlim())
    axw.set_ylim(*ax.get_ylim())
    axw.set_axis_off()

    # plot windbarbs
    axw.barbs(np.full(len(ax.data['P']),-16.),ax.data['P'],ax.data['U']*1.94,ax.data['V']*1.94,
              length=6  ,linewidth=0.5,zorder=10)

    #- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 
    print('Surface parcel')
    parcel0 = ax.data['parcel']
    ax.plot(parcel0['T'],parcel0['P'],color='black',linewidth=2.0)
    for key in ['LCL','CCL','LFC','EL']:
      print(f'  {key}=',parcel0[key]) 
      ax.plot(parcel0[key]['T'],parcel0[key]['P'],'*',color='purple',ms=7,zorder=20)

    print('  CIN:',parcel0['CIN'].keys())
    print('  CIN=',parcel0['CIN']['value']) 
    if parcel0['CIN']['value']!=0.:
      ax.fill_betweenx(parcel0['CIN']['P'],parcel0['CIN']['Ta'],parcel0['CIN']['Te'],
                       facecolor='blue',alpha=0.3,zorder=2)

    print('  CAPE:',parcel0['CAPE'].keys())
    print('  CAPE=',parcel0['CAPE']['value'])
    if parcel0['CAPE']['value']!=0.:
      ax.fill_betweenx(parcel0['CAPE']['P'],parcel0['CAPE']['Ta'],parcel0['CAPE']['Te'],
                       facecolor='red',alpha=0.3,zorder=2)
    #- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 
    print('Specific parcel')
    parcel1 = ax.data.get_parcel(925.)
    ax.plot(parcel1['T'],parcel1['P'],color='cyan',linewidth=2.)
    for key in ['LCL','CCL','LFC','EL']:
      print(f'  {key}=',parcel1[key])
    print('  CIN=' ,parcel1['CIN']['value'])
    print('  CAPE=',parcel1['CAPE']['value'])
    #- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 

    plt.show()


