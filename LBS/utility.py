import os
import cv2
import numpy as np
from scipy.spatial.distance import pdist, squareform
from sklearn.decomposition import PCA
from PIL import Image
import matplotlib
matplotlib.use("Agg")
from matplotlib import pyplot, cm
from mpl_toolkits.mplot3d import Axes3D

def read_image(true_field, n):
    image = cv2.imread(true_field+'.png',0)
    imagesmall = cv2.resize(image,(n-1,n-1))
    imagesmallfloat = imagesmall.astype(float)
    np.place(imagesmallfloat, imagesmallfloat == 0, 1.5)
    np.place(imagesmallfloat, imagesmallfloat > 1.5, 1)
    synthetic_u = np.log(imagesmallfloat).reshape(-1)
    return synthetic_u

def construct_ob(true_solu, Sscale, n, SNR, manualSeed, standardob, symmetricob = False):
    np.random.seed(manualSeed)
    if symmetricob:
        if standardob:
            ob_pos_x,ob_pos_y = np.linspace(0,1,n//4+1), np.ones(n//4+1)*0.5
        else:
            ob_pos_x,ob_pos_y = np.linspace(0,1,n), np.ones(n)*0.5
    else:
        if standardob:
            ob_pos_x,ob_pos_y = np.meshgrid(np.linspace(0.2,0.6,5), np.linspace(0.2,0.6,5))
        else:
            ob_pos_x,ob_pos_y = np.meshgrid(np.linspace(0,1,n), np.linspace(0,1,n))
    ob_pos = np.concatenate((ob_pos_x.reshape((1,-1)),ob_pos_y.reshape((1,-1))),axis=0)
    ob_idx = np.round(ob_pos[0,:]*(n-1)*n+ob_pos[1,:]*(n-1)).astype(int)
    ob_data = true_solu[0, ob_idx]
    noise = np.random.normal(0, Sscale/SNR, ob_data.shape)
    ob_data = ob_data + noise
    observation = np.row_stack((ob_idx,ob_data))
    return observation, noise

def findmap(u0, lnprob):
    umap = u0.copy()
    lnprob_now, dlnprob_now = lnprob(umap)[1:]
    stepsize = 1e-1
    for i in range(100):
        u1 = umap + stepsize*dlnprob_now
        lnprob1, dlnprob1 = lnprob(u1)[1:]
        if lnprob1 < lnprob_now:
            while lnprob1 < lnprob_now:
                stepsize = stepsize / 2
                u1 = umap + stepsize*dlnprob_now
                lnprob1, dlnprob1 = lnprob(u1)[1:]
            umap = u1.copy()
            lnprob_now = lnprob1
            dlnprob_now = dlnprob1.copy()
        else:
            stepsize = stepsize * 2
            u2 = umap + stepsize*dlnprob_now
            lnprob2_old = lnprob1
            dlnprob2_old = dlnprob1.copy()
            lnprob2, dlnprob2 = lnprob(u2)[1:]
            while lnprob2 > lnprob1:
                lnprob1 = lnprob2
                stepsize = stepsize * 2
                u2= umap + stepsize*dlnprob_now
                lnprob2_old = lnprob2
                dlnprob2_old = dlnprob2.copy()
                lnprob2, dlnprob2 = lnprob(u2)[1:]
            stepsize = stepsize / 2
            umap = umap + stepsize*dlnprob_now
            lnprob_now = lnprob2_old
            dlnprob_now = dlnprob2_old.copy()
    return umap

class ImageGenerator():
    
    def __init__(self, n, true_u, foldername):
        self.n = n
        self.foldername = foldername
        self.xu3d, self.yu3d = (np.arange(n-1)+0.5)/(n-1), (np.arange(n-1)+0.5)/(n-1)
        self.xu, self.yu = np.arange(n)/(n-1), np.arange(n)/(n-1)
        self.my_dir = os.getcwd()
        self.true_u = true_u.reshape((self.n-1,self.n-1))
        self.vmax = np.max(self.true_u)+0.25
        self.vmin = np.min(self.true_u)-0.25
        
    def training(self, u_sample, iteration, nlnp, lnp = None):
        fig = pyplot.figure()
        ax = fig.gca(projection='3d')
        ax.set_zlim3d(self.vmin, self.vmax)
        ax.plot_surface(self.xu3d,self.yu3d,self.true_u.T,cmap=cm.coolwarm,linewidth=0,antialiased=False,alpha=0.25)
        surf = ax.plot_surface(self.xu3d,self.yu3d,u_sample.T,cmap=cm.coolwarm,linewidth=0,antialiased=False,alpha=0.75)
        pyplot.xlabel('x')
        pyplot.ylabel('y')
        ax.view_init(azim=315)
        pyplot.colorbar(surf)
        pyplot.title('Iteration '+format(iteration)+': u plot, -log(p)='+format(nlnp,'3.2e'))
        pyplot.savefig(os.path.join(self.my_dir, 'Data', self.foldername, 'u_plot_iter'+format(iteration)+'.png'))
        pyplot.close()
        
        pyplot.figure()
        pyplot.contour(self.xu3d,self.yu3d,self.true_u)
        pyplot.pcolor(self.xu,self.yu,u_sample.T,vmin=self.vmin, vmax=self.vmax)
        pyplot.colorbar()
        pyplot.title('Iteration '+format(iteration)+': u contour, -log(p)='+format(nlnp,'3.2e'))
        pyplot.savefig(os.path.join(self.my_dir, 'Data', self.foldername, 'u_cont_iter'+format(iteration)+'.png'))
        pyplot.close()
        
        if lnp is not None:
            pyplot.figure()
            pyplot.plot(lnp)
            pyplot.title('Iteration '+format(iteration)+': lnprob')
            pyplot.savefig(os.path.join(self.my_dir, 'Data', self.foldername, 'u_lnp_iter'+format(iteration)+'.png'))
            pyplot.close()
        
    def conclude(self, samples, samples_mean, samples_var, samples_lnp, prefix = ''):
        D = squareform(pdist(samples, metric='euclidean'))
        pca = PCA(n_components=2)
        samples_pca = pca.fit_transform(samples)
        
        pyplot.figure()
        pyplot.pcolor(D)
        pyplot.colorbar()
        pyplot.title('Pairwise Distances of Samples')
        pyplot.savefig(os.path.join(self.my_dir, 'Data', self.foldername, prefix+'sample_pair_dists.png'))
        pyplot.close()
        
        pyplot.figure()
        pyplot.scatter(samples_pca[:,0], samples_pca[:,1], marker = '.')
        pyplot.title('PCA of samples')
        pyplot.savefig(os.path.join(self.my_dir, 'Data', self.foldername, prefix+'sample_pca_2d.png'))
        pyplot.close()
        
        if samples_mean is not None:
            fig = pyplot.figure()
            ax = fig.gca(projection='3d')
            ax.plot_surface(self.xu3d,self.yu3d,self.true_u.T,cmap=cm.coolwarm,linewidth=0,antialiased=False,alpha=0.7)
            surf = ax.plot_surface(self.xu3d,self.yu3d,samples_mean.reshape((self.n-1, self.n-1)).T,cmap=cm.coolwarm,linewidth=0,antialiased=False,alpha=0.2)
            pyplot.xlabel('x')
            pyplot.ylabel('y')
            ax.view_init(azim=315)
            pyplot.colorbar(surf)
            pyplot.title('sample mean')
            pyplot.savefig(os.path.join(self.my_dir, 'Data', self.foldername, prefix+'sample_mean_3d.png'))
            pyplot.close()
                
            pyplot.figure()
            pyplot.pcolor(self.xu, self.yu, samples_mean.reshape((self.n-1, self.n-1)).T)
            pyplot.colorbar()
            pyplot.title('sample mean')
            pyplot.savefig(os.path.join(self.my_dir, 'Data', self.foldername, prefix+'sample_mean.png'))
            pyplot.close()
        
        if samples_var is not None:
            pyplot.figure()
            pyplot.pcolor(self.xu, self.yu, samples_var.reshape((self.n-1, self.n-1)).T)
            pyplot.colorbar()
            pyplot.title('sample var')
            pyplot.savefig(os.path.join(self.my_dir, 'Data', self.foldername, prefix+'sample_var.png'))
            pyplot.close()
        
        if samples_lnp is not None:
            pyplot.figure()
            pyplot.plot(samples_lnp)
            pyplot.title('sample lnprob')
            pyplot.savefig(os.path.join(self.my_dir, 'Data', self.foldername, prefix+'sample_lnprob.png'))
            pyplot.close()
        
    def conclude_burnin(self, burnin_lnp):
        pyplot.figure()
        pyplot.plot(burnin_lnp)
        pyplot.title('sample burn-in lnprob')
        pyplot.savefig(os.path.join(self.my_dir, 'Data', self.foldername, 'sample_burn-in_lnprob.png'))
        pyplot.close()
        
    def training_curve(self, train_curve_data):
        pyplot.figure()
        pyplot.semilogy(np.median(train_curve_data, axis = 0), color = 'k', label = 'Median', alpha = 1.0)
        pyplot.semilogy(np.max(train_curve_data, axis = 0), color = 'r', label = 'Maximum', alpha = 0.4)
        pyplot.semilogy(np.min(train_curve_data, axis = 0), color = 'b', label = 'Minimum', alpha = 0.4)
        pyplot.legend(loc = 'upper right')
        pyplot.title('Training Curve of Negative Log of Posterier Probability')
        pyplot.savefig(os.path.join(os.getcwd(), 'Data', self.foldername, 'training_curve_'+self.foldername+'.png'))
        pyplot.close()
        np.savez(os.path.join(self.my_dir, 'Data', self.foldername, 'nlnprob.npz'), nlnprob_train = train_curve_data)
        
        
    def conclude_make_gif(self, niter, check_period):
        my_dir = os.getcwd()
        im = Image.open(os.path.join(self.my_dir, 'Data', self.foldername, 'u_plot_iter'+format(check_period)+'.png'))    
        images=[]
        for i in range(niter//check_period-1):
            images.append(Image.open(os.path.join(self.my_dir, 'Data', self.foldername, 'u_plot_iter'+format((i+2)*check_period)+'.png')))
        im.save(os.path.join(self.my_dir, 'Data', self.foldername, 'demo_u_plot_'+self.foldername+'.gif'), save_all=True, append_images=images, loop=1, duration=400, comment=b"aaabb")
        
        im_cont = Image.open(os.path.join(self.my_dir, 'Data', self.foldername, 'u_cont_iter'+format(check_period)+'.png'))
        images=[]
        for i in range(niter//check_period-1):
            images.append(Image.open(os.path.join(my_dir, 'Data', self.foldername, 'u_cont_iter'+format((i+2)*check_period)+'.png')))
        im_cont.save(os.path.join(self.my_dir, 'Data', self.foldername, 'demo_u_cont_'+self.foldername+'.gif'), save_all=True, append_images=images, loop=1, duration=400, comment=b"aaabb")