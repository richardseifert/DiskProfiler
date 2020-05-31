import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from scipy.interpolate import griddata,interp2d
from astropy.io import fits
from astropy.wcs import WCS

class profiler:
    def __init__(self,fpath,mpath=None,cx=None,cy=None,pa=None,inc=None,**kwargs):
        #Load data as fitscube object.
        self.cube = fitscube(fpath=fpath,mpath=mpath,**kwargs)
        #Set disk geometry.
        self.geom = diskgeom(self.cube,cx=cx,cy=cy,pa=pa,inc=inc)
        
        #Dictionaries for storing intermediate products.
        self.points = {}
        self.values = {}

        #Dictionaries for unit conversions
        self.sunit_conv = {'deg':1.0,'arcmin':60.0,'arcsec':3600.0}
        self.bunit_conv = {'Jy/beam':1.0,'mJy/beam':1000}

    def get_points(self,spat='radec'):
        k = 'spat_%s'%(spat)
        if not k in self.points.keys():
            r,az = self.geom.get_raz_arrs(use=spat)
            self.points[k] = np.c_[r.flatten(),az.flatten()]
        return self.points[k].copy()
    def get_values(self):
        k = 'mom0'
        if not k in self.values.keys():
            self.values[k] = self.cube.get_mom0().flatten()
        return self.values[k].copy()

    def get_profile(self,along='r',rlo=0,rhi=None,azlo=0,azhi=360,nbins=100,dx=None,spat='radec',spat_unit='arcsec',bunit='mJy/beam',noise_method=None):
        #Grab r and az 1D arrays.
        rpts,azpts = self.get_points(spat).T
        rpts *= self.sunit_conv[spat_unit]

        #Handle input r and az bounds: rlo,rhi, azlo,azhi
        if rhi is None:
            rhi = np.max(rpts)
        daz = (azhi-azlo)%360
        if daz == 0: daz = 360
        az_offset = azlo
        azpts = (azpts-azlo)%360
        azlo = 0
        azhi = daz

        #Generate x array.
        if along == 'r':
            xpts = rpts
            Dx = rhi-rlo
            xlo = rlo
            xhi = rhi
        elif along == 'az':
            xpts = azpts
            Dx = daz
            xlo = azlo
            xhi = azhi
        if not dx is None:
            nbins = Dx/dx + 1
        x = np.linspace(xlo,xhi,nbins)

        #Grab mom0 brightness values.
        bpts = self.get_values()
        bpts *= self.bunit_conv[bunit]

        #Mask according to non-along axis
        mask = (azpts >= azlo) & (azpts <= azhi) & (rpts >= rlo) & (rpts <= rhi)
        bpts = bpts[mask]
        xpts = xpts[mask]

        #Split into bins
        bpts_binned = binary_chunkify(bpts,bins=x,barr=xpts)

        #Average
        y = np.array([np.mean(bpts) for bpts in bpts_binned])

        if noise_method == 'std':
            dy = np.array([np.std(bpts) for bpts in bpts_binned])
        elif noise_method == 'Nbeam':
            #dy = npts*pixarea/beamarea * noise_per_beam
            pass
        elif noise_method is None:
            dy = np.zeros_like(y)

        #Return
        return x,y,dy

    def plot_profile(self,ax=None,kind='smooth',plot_kwargs={},fill_kwargs={},**profile_kwargs):
        if ax is None:
            fig,ax = plt.subplots()
        x,y,dy = self.get_profile(**profile_kwargs)
        
        #Plot error
        fkwargs = {'color':'cornflowerblue','alpha':0.8,'linewidth':0} #Default fill kwargs
        if kind == 'step': fkwargs['step'] = 'mid'
        fkwargs.update(fill_kwargs)
        ax.fill_between(x,y-dy,y+dy,**fkwargs)
        #Plot profile
        pkwargs = {'color':'black'} #Default plot kwargs
        if kind == 'step': pkwargs['where'] = 'mid'
        pkwargs.update(plot_kwargs)
        if kind == 'smooth':
            ax.plot(x,y,**pkwargs)
        elif kind == 'step':
            ax.step(x,y,**pkwargs)

        return ax

    def plot_summarized_profile(self,along='r',rlo=0,rhi=None,azlo=0,azhi=360,img_ax=None,prf_ax=None,disp_img=True,**kwargs):
        #Plot image and grid on img_ax

        #Plot profile on prf_ax

        return img_ax,prf_ax

    def get_segmented_rprofs(self,rlo=0,rhi=None,nbins=100,dr=None,azlo=0,azhi=360,nseg=8,spat='radec'):
        daz = (azhi-azlo)%360
        if daz == 0:
            daz = 360
        bins = (np.arange(nseg+1)*daz/(nseg) + azlo)%360
        rprofs = {}
        for azl,azh in zip(bins[:-1],bins[1:]):
            R,I = self.get_rprofile(rlo=rlo,rhi=rhi,nbins=nbins,dr=dr,azlo=azl,azhi=azh,spat=spat)
            rprofs[0.5*(azl+azh)] = I
        rprofs['R'] = R
        return rprofs

    ### Re-route to fitscube methods ###
    def display(self,center=True,spat_unit='arcsec',bunit='mJy/beam',*args,**kwargs):
        xarr,yarr = self.geom.get_radec_arrs(center=center)
        xarr *= self.sunit_conv[spat_unit]
        yarr *= self.sunit_conv[spat_unit] #Convert from, e.g., degrees to arcseconds.
        return self.cube.display(xarr=xarr,yarr=yarr,*args,**kwargs)

    ### Re-route to diskgeom methods ###
    def plot_ellipse(self,*args,**kwargs):
        return self.geom.plot_ellipse(*args,**kwargs)
    def plot_ray(self,*args,**kwargs):
        return self.geom.plot_ray(*args,**kwargs)


class fitscube:
    def __init__(self,fpath,mpath=None,xi=None,yi=None,vi=None,dvel=None):
        #Load image and mask
        self.img,self.head = self.load_cube(fpath,header=True)
        if not mpath is None:
            self.mask = self.load_cube(mpath)
            if not np.all(self.mask.shape == self.img.shape):
                print("Warning: Mask provided has incompatible shape! Not loading.")
                mpath = None
        if mpath is None:
            self.mask = np.ones_like(self.img)

        self.set_axes(xi=xi,yi=yi,vi=vi) #Look in header for x,y,v axes indices.
        self.set_dvel(dvel=dvel) #Look in header to get dvel.

        self.init_wcs()

    def init_wcs(self):
        ra_n = self.header_get_CN(look_for='RA')
        dec_n = self.header_get_CN(look_for='DEC')
        self.w = WCS(naxis=2)
        self.w.wcs.crpix = [self.head['CRPIX%d'%(n)] for n in [ra_n,dec_n]]
        self.w.wcs.cdelt = [self.head['CDELT%d'%(n)] for n in [ra_n,dec_n]]
        self.w.wcs.crval = [self.head['CRVAL%d'%(n)] for n in [ra_n,dec_n]]
        self.w.wcs.ctype = [self.head['CTYPE%d'%(n)] for n in [ra_n,dec_n]]
    def pix2world(self,x,y):
        return self.w.wcs_pix2world(x,y,0)
    def world2pix(self,ra,dec):
        return self.w.wcs_world2pix(ra,dec,0)

    def get_xy_arrs(self):
        x1d = np.arange(self.get_nx())
        y1d = np.arange(self.get_ny())
        return np.meshgrid(x1d,y1d)
    def get_radec_arrs(self):
        x,y = self.get_xy_arrs()
        return self.pix2world(x,y)


    def load_cube(self,path,header=False,trim=True,transpose=True):
        '''
        Load fits file (hopefully 3-dimensional). Optionally trim dimensions of size 1.

        ARGUMENTS:
            path - String path to an existing fits file.
            trim - Boolean whether or not to trim empty dimensions. Default True
        RETURNS:
            dat  - Numpy array of loaded fits file, possibly trimmed.
        '''
        #Load from file
        f = fits.open(path)
        dat = f[0].data
        if transpose:
            dat = dat.T #Transpose axes so they match fits header.
        if header:
            head = f[0].header
        f.close()

        #Trim dimensions of size 1.
        if trim:
            indx = tuple([slice(None) if dat.shape[i]>1 else 0 for i in range(dat.ndim)])
            dat = dat[indx]

        #Return!
        if header:
            return dat,head
        else:
            return dat

    def get_nx(self):
        return self.img.shape[self.xi]
    def get_ny(self):
        return self.img.shape[self.yi]
    def get_nchan(self):
        if self.vi is None:
            return 1
        return self.img.shape[self.vi]

    def get_mom0(self,use_mask=True,deproj=False):
        if self.vi is None:
            return self.img

        nchan = self.get_nchan()
        specarr = self.dvel*np.arange(nchan)
        if use_mask:
            cube = self.img*self.mask
        else:
            cube = self.img
        mom0 = np.trapz(np.moveaxis(cube,[self.xi,self.yi,self.vi],[0,1,2]),x=specarr,axis=2).T
        if deproj:
            pass
        else:
            return mom0

    def header_get_CN(self,look_for,get='first'):
        '''
        Get axis number of fits axis whose CTYPE contrains a given string.

        ARGUMENTS:
            look_for - String or list of strings to look for in CTYPEs in fits header.
            get      - Method for returning matches:
                         'first' - Default. Return first match.
                         'all'   - Return list of all matches.
        RETURNS:
            good_N   - 1-index index or list of indices for matching axis(es).
        '''
        if isinstance(look_for,str):
            look_for = [look_for]
        good_n = []
        found_it = False
        for n in range(1,self.head['NAXIS']+1):
            for term in look_for:
                if term in self.head['CTYPE%d'%(n)]:
                    found_it = True
            if found_it:
                if get=='first':
                    return n
                else:
                    good_n.append(n)
                    found_it = False
        if len(good_n) == 0:
            return None
        return good_n
        
    def set_axes(self,xi=None,yi=None,vi=None):
        '''
        Set spatial and spectral axes. If not provided, they will
        be found from the image header.
        '''
        if xi is None or yi is None or vi is None:
            #Get indices from image header.
            found = self.find_axes()

        #Set x spatial axis.
        if not xi is None:
            self.xi = xi
        else:
            if found['xi'] is None:
                raise ValueError("Could not determine which axes corresponds to RA")
            self.xi = found['xi']

        #Set y spatial axis.
        if not yi is None:
            self.yi = yi
        else:
            if found['yi'] is None:
                raise ValueError("Could not determine which axes corresponds to DEC")
            self.yi = found['yi']

        #Set v spectral axis.
        if not vi is None:
            self.vi = vi
        else:
            if found['vi'] is None:
                raise ValueError("Could not determine which axes corresponds to Frequency/Velocity")
            self.vi = found['vi']
            if self.vi >= self.img.ndim:
                self.vi = None
    def find_axes(self): 
        '''
        Use image header to determine axes indices corresponding to spatial and spectral axes.
        '''
        search_terms = {'xi':['RA'],'yi':['DEC'],'vi':['FREQ','VEL']}
        indices = {k:None for k in search_terms.keys()}
        for k in indices.keys():
            indices[k] = self.header_get_CN(search_terms[k])
            if not indices[k] is None:
                indices[k] -= 1 #Go from 1-index fits indexing to 0-index numpy indexing
        return indices

    def set_dvel(self,dvel=None):
        '''
        Set channel velocity width. If not provided, it will be found from
        the image header. 
        '''
        if dvel is None:
            #Compute value from header.
            self.dvel = self.find_dvel()
        else:
            self.dvel = dvel
    def find_dvel(self):
        '''
        Use image header to determine channel width.
        '''
        N = self.header_get_CN(['FREQ','VEL'])
        return np.abs(self.head['CDELT%d'%(N)] / self.head['CRVAL%d'%(N)] * 3e5) # Velocity res in km/s
    
    def display(self,img=None,method='contour',spat='radec',xarr=None,yarr=None,norm='linear',vmin=None,
            vmax=None,levels=25,cmap='viridis',cbar=True,cbar_ax=None,ax=None,\
            xlim=None,ylim=None,fill=True):
        #Handle inputs!
        if img is None:
            img = self.get_mom0()
        # make axes unless one is given.
        if ax is None:
            fig,ax = plt.subplots()
        # determine scale limits, if not given.
        if vmin is None:
            vmin = np.nanmin(img)
            if norm == 'log' and vmin <= 0:
                vmin = 1e-20
        if vmax is None:
            vmax = np.nanmax(img)
        
        if xarr is None and spat == 'radec':
            xarr,yarr = self.get_radec_arrs()
        if xarr is None and spat == 'pix':
            xarr,yarr = self.get_xy_arrs()

        if norm=='linear':
            cmnorm = None
        elif norm=='log':
            cmnorm = LogNorm(vmin=np.log10(vmin),vmax=np.log10(vmax))
            img[img<=0] = vmin

        if method == 'contour':
            #Preparations:
            # if levels was given as int, make array levels.
            try:
                iter(levels)
            except TypeError:
                #It's scalar! Make it a vector.
                if norm=='linear':
                    levels = np.linspace(vmin,vmax,levels)
                elif norm=='log':
                    levels = np.geomspace(vmin,vmax,levels)
            #Plot!
            if fill:
                im = ax.contourf(xarr,yarr,img,levels=levels,cmap=cmap,extend='both',norm=cmnorm)
            else:
                im = ax.contour(xarr,yarr,img,levels=levels,cmap=cmap,extend='both',norm=cmnorm)

        if method == 'imshow':
            #Preparations:
            # determine ra and dec bounds.
            nx = img.shape[0]
            ny = img.shape[1]
            x_bl = xarr[nx-1,0] #bottom left
            x_tr = xarr[0,ny-1] #top right
            y_bl = yarr[nx-1,0]
            y_tr = yarr[0,ny-1]
            #Plot!
            im = ax.imshow(img,cmap=cmap,vmin=vmin,vmax=vmax,extent=[x_bl,x_tr,y_bl,y_tr],norm=cmnorm)

        #Default, RA right->left, DEC bottom->top
        xleft,xright=ax.get_xlim()
        if xright > xleft: ax.set_xlim(xright,xleft)
        ybottom,ytop=ax.get_ylim()
        if ybottom > ytop: ax.set_ylim(ytop,ybottom)

        #If provided, set xlim and ylim to user specified.
        try:
            iter(xlim)
            if len(xlim) >= 2:
                ax.set_xlim(*xlim[:2])
        except TypeError:
            pass
        try:
            iter(ylim)
            if len(ylim) >= 2:
                ax.set_ylim(*ylim[:2])
        except TypeError:
            pass

        if cbar:
            cax = self._make_cbar(ax,im,cbar_ax)
            return ax,cax
        return ax,None

    def _make_cbar(self,ax, im, cbar_ax):
        if cbar_ax is None:
            cb=ax.get_figure().colorbar(im)
        else:
            cb=ax.get_figure().colorbar(im, cax=cbar_ax)
        return cb.ax

class diskgeom:
    def __init__(self,cube,cx=None,cy=None,pa=0,inc=0):
        self.cube = cube

        cx = self.cube.get_nx()//2
        cy = self.cube.get_ny()//2
        cra, cdec = self.cube.pix2world(cx,cy)
        self.g = {'cra':cra,'cdec':cdec,'pa':0,'inc':0}
        self.geom_set = {k:False for k in self.g.keys()}
        self.set_geometry(cra=cra,cdec=cdec,pa=pa,inc=inc)

        #Dicts to store griddata inputs.
        self.points = {}
        self.values = {}

    def get(self,k):
        return self.g[k]

    def set_geometry(self,**kwargs):
        '''
        Set geometric quantities.

        ARGUMENTS:
          If any are not provided, they will not be set.
            cx  - x coordinate of disk center on the provided image. Default is center of image.
            cy  - y coordinate of disk center on the provided image. Default is center of image.
            pa  - Position angle of disk, in degrees.
            inc - Inclination of disk, in degrees.
        RETURNS:
            Nothing. Variables are set.
        '''
        for k in self.g.keys():
            if k in kwargs and not kwargs[k] is None:
                self.g[k] = kwargs[k]
                self.geom_set[k] = True
    def _warn_geometry(self):
        '''
        Issue warning to the user in the event that some geometric quantities are not explicitly set.
        '''
        unset = {}
        for k in self.g.keys():
            if not self.geom_set[k]:
                unset[k] = self.g[k] 

        if len(unset) > 0:
            print("Warning: Some parameters have not been explicitly set. Using Defaults:")
            for k,v in unset.items():
                print("\tUsing %s = %.2f"%(k,v))
        
    def get_xy_arrs(self,center=True):
        '''
        Get x and y arrays with same shape as 2D image
        '''
        x,y = self.cube.get_xy_arrs()
        if center:
            cx,cy = self.cube.world2pix(self.g['cra'],self.g['cdec'])
            return x-cx, y-cy
        else:
            return x,y
    def get_radec_arrs(self,center=True):
        ra,dec = self.cube.get_radec_arrs()
        if center:
            return ra-self.g['cra'], dec-self.g['cdec']
        else:
            return ra, dec
    def get_raz_arrs(self,use='radec'):
        '''
        Get radius and azimuth arrays with same shape as 2D image
        '''
        pa = self.g['pa']*np.pi/180.
        inc= self.g['inc']*np.pi/180.
        if use == 'radec':
            ra,dec = self.get_radec_arrs()
            phi = np.arctan2(dec,ra)
            d = (ra**2+dec**2)**0.5
        elif use == 'xy':
            x,y = self.get_xy_arrs()
            phi = np.arctan2(y,x)
            d = (x**2+y**2)**0.5
        e = (1-np.cos(inc)**2)**0.5
        b = d*(1-e*np.cos(phi-pa)**2)**0.5
        r = b/np.cos(inc)
        az = (phi*180/np.pi+180-self.g['pa'])%360
        return r,az

    def get_grid(self,inputs="rphi",target="mom0"):
        #Get griddata points and values
        return self.get_points(inputs), self.get_values(target)
    def get_points(self,k):
        if k == 'raz':
            if not k in self.points.keys():
                r,az = self.get_raz_arrs()
                self.points[k] = np.c_[r.flatten(),az.flatten()]
            return self.points[k]
        if k == 'xy':
            if not k in self.points.keys():
                x,y = self.get_xy_arrs()
                self.points[k] = np.c_[x.flatten(),y.flatten()]
            return self.points[k]
        raise ValueError("Unknown griddata points key, %s"%(k))
    def get_values(self,k):
        if k == 'mom0':
            if not k in self.values.keys():
                mom0 = self.cube.get_mom0()
                self.values['mom0'] = mom0.flatten()
            return self.values['mom0']
        if k == 'r':
            pts = self.get_points('raz')
            return pts[:,0]
        if k == 'az':
            pts = self.get_points('raz')
            return pts[:,1]
        if k == 'x':
            pts = self.get_points('xy')
            return pts[:,0]
        if k == 'y':
            pts = self.get_points('xy')
            return pts[:,1]
        raise ValueError("Unknown griddata values key, %s"%(k))

    def get_raz_mask(self,rlo=0,rhi=None,azlo=0,azhi=360,use='radec'):
        r,az = self.get_raz_arrs(use=use)
        if rhi is None:
            rhi = np.max(r)
        az = (az-azlo)%360
        daz = (azhi-azlo)%360
        if daz == 0: daz = 360
        return (r >= rlo) & (r <= rhi) & (az >= 0) & (az <= daz)

    def deproj(self,img=None):
        if img is None:
            img = self.cube.get_mom0()
        #Interpolate brightness over r,phi
        x,y = self.get_xy_arrs()
        r,az = self.get_raz_arrs(use='xy')
        d = (x**2+y**2)**0.5
        points = np.c_[r.flatten(),az.flatten()]
        values = img.flatten()
        deproj = griddata(points, values, (d,az),method='linear')
        deproj[np.isnan(deproj)] = 0.0
        return deproj
    def plot_center(ax=None,center=True,**scatter_kwargs):
        if ax is None:
            fig,ax = plt.subplots()
        if not center:
            cx = self.g['cx']
            cy = self.g['cy']
        else:
            cx,cy = 0.,0.
        ax.scatter([cx],[cy],**scatter_kwargs)
        
    def plot_ellipse(self,rad,azlo=0,azhi=360,use='radec',center=True,ax=None,**contour_kwargs):
        if ax is None:
            fig,ax = plt.subplots()
        try:
            iter(rad)
        except TypeError:
            rad = [rad]
        rad = np.sort(rad)

        r,az = self.get_raz_arrs(use)
        az = (az-azlo)%360
        daz = (azhi-azlo)%360
        if daz == 0: daz = 360
        mr = r.copy()
        mr[az > daz] = np.nan

        if use == 'radec':
            xarr,yarr = self.get_radec_arrs(center)
        elif use == 'xy':
            xarr,yarr = self.get_xy_arrs(center)
        ax.contour(xarr,yarr,mr,levels=rad,**contour_kwargs)

    def plot_ray(self,azim,rlo=0,rhi=None,npts=100,use='radec',center=True,ax=None,**contour_kwargs):
        if ax is None:
            fig,ax = plt.subplots()
        try:
            iter(azim)
        except TypeError:
            azim = [azim]
        azim = np.sort(azim)

        if rhi is None:
            rhi = self.cube.get_nx()//2

        r,az = self.get_raz_arrs(use)
        x,y   = self.get_xy_arrs()
        maz = az.copy()
        
        #Shift az = 0 to the most distant angle from an azim to be plotted. 
        dists = [(azi1-azi2)%360 for azi1,azi2 in zip(azim,np.roll(azim,1))]
        mdi = np.argmax(dists)
        md = dists[mdi]
        zangle = (azim[mdi]+md/2)%360
        maz = (maz - zangle)%360
        azim = np.sort([(azi-zangle)%360 for azi in azim])

        maz[(r < rlo) | (r > rhi) | (180-np.abs(180-maz) < md/5) ] = np.nan
        

        if use == 'radec':
            xarr,yarr = self.get_radec_arrs(center)
        elif use == 'xy':
            xarr,yarr = self.get_xy_arrs(center)
        ax.contour(xarr,yarr,maz,levels=azim,**contour_kwargs)

    def plot_grid(self,rlo,rhi,azlo=0,azhi=360,Nr=10,Naz=10,mark_center=True,color='blue',**contour_kwargs):
        pass


def binary_chunkify(arr,bins,barr=None):
    if len(bins) == 1:
        split = barr<=bins[0]
        chunks = [arr[split],arr[~split]]
        return chunks
    elif len(bins) == 0:
        return [arr]
    else:
        chunks = []
        i = int(len(bins)/2)
        split = barr<=bins[i]
        left_chunks = binary_chunkify(arr[split],bins[:i],barr=barr[split])
        right_chunks = binary_chunkify(arr[~split],bins[i+1:],barr=barr[~split])
        chunks.extend(left_chunks)
        chunks.extend(right_chunks)
        return [chunk for chunk in chunks if len(chunk) > 0]
