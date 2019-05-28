import numpy as np
from scipy.sparse import coo_matrix
from scipy.sparse.linalg import spsolve
from scipy.spatial.distance import pdist, squareform

class pde_adj_solver():
    
    def __init__(self, n, Sscale, SNR, force, lnprob_likelihood=False):  
        self.lnprob_likelihood = lnprob_likelihood
        
        self.sigma = Sscale/SNR
        self.n = n
        self.nbdy = 4*(self.n-1)
        self.face, self.vertex, self.bdy = self.meshgen_2dsq()
        
        if not self.lnprob_likelihood:
            self.sigmau = 1.25
            self.s0 = 0.0625
            x,y = np.meshgrid(np.arange(0,1,1/(n-1)),np.arange(0,1,1/(n-1)))
            X = np.concatenate((x.reshape(1, -1),y.reshape(1, -1)),axis = 0).transpose()
            d = squareform(pdist(X, metric='euclidean'))
            self.Gamma = self.sigmau**2*np.exp(-1/(2*self.s0)*d)
            self.inv_Gamma = np.linalg.inv(self.Gamma)
        
        nodes = self.vertex[self.face[0,:],:]
        D_Phi = np.transpose(0.5*np.column_stack((nodes[1,:]-nodes[0,:],nodes[3,:]-nodes[0,:])))
        det_D_Phi = -np.linalg.det(D_Phi)
        B = np.linalg.inv(np.matmul(np.transpose(D_Phi),D_Phi))
        
        self.gradphisq2 = np.zeros(16,dtype=float)
        self.gradphisq2[0] = 16/3*(B[0,0] + B[1,1]) + 8*B[0,1]
        self.gradphisq2[1] = 8/3*(B[1,1] - 2*B[0,0])
        self.gradphisq2[2] = -8/3*(B[0,0] + B[1,1]) - 8*B[0,1]
        self.gradphisq2[3] = 8/3*(B[0,0] - 2*B[1,1])
        self.gradphisq2[4] = self.gradphisq2[1]
        self.gradphisq2[5] = 16/3*(B[0,0] + B[1,1]) - 8*B[0,1]
        self.gradphisq2[6] = 8/3*(B[0,0] - 2*B[1,1])
        self.gradphisq2[7] = 8*B[0,1] - 8/3*(B[0,0]+B[1,1])
        self.gradphisq2[8] = self.gradphisq2[2]
        self.gradphisq2[9] = self.gradphisq2[6]
        self.gradphisq2[10] = 16/3*(B[0,0] + B[1,1]) + 8*B[0,1]
        self.gradphisq2[11] = 8/3*(B[1,1] - 2*B[0,0])
        self.gradphisq2[12] = self.gradphisq2[3]
        self.gradphisq2[13] = self.gradphisq2[7]
        self.gradphisq2[14] = self.gradphisq2[11]
        self.gradphisq2[15] = 16/3*(B[0,0] + B[1,1]) - 8*B[0,1]
        
        self.gradphisq2 = self.gradphisq2 * det_D_Phi / 16
        self.b = np.zeros(self.n**2,dtype=float)
        
        for j in range(0, (self.n-1)**2):
            nodes = self.vertex[self.face[j,:],:]
            center_force = force(np.mean(nodes[:,0]),np.mean(nodes[:,1]))
            self.b[self.face[j,:]] = self.b[self.face[j,:]] + np.ones(4) * det_D_Phi * center_force
            
        self.I = np.matlib.repmat(self.face.flatten(),4,1).flatten('F')
        self.J = np.matlib.repmat(self.face,1,4).flatten()
        # Modify Stiffness Matrix according to Boundary Conditions, for neumann condition, introduce normalization
        self.idx = self.n//2*self.n+self.n//2
        self.idx_in = np.where(self.I != self.idx)[0]
        self.Ibdy = np.concatenate((self.I[self.idx_in], np.ones(self.nbdy)*self.idx))
        self.Jbdy = np.concatenate((self.J[self.idx_in], self.bdy[:,0]))
        self.Vphi = np.matlib.repmat(self.gradphisq2, (self.n-1)**2, 1)
    
    def solve(self, u):
        if u.ndim is 1:
            u = u[np.newaxis,:]
        batchsize = u.shape[0]
        u = np.reshape(u, (batchsize, -1))
        kappa = np.exp(u)
        
        nvert = self.n**2
        p = np.zeros((batchsize, nvert), dtype=float)
        
        b = self.b.copy()
        b[self.idx] = 0
        
        for i in range(batchsize):
            Vfull = (self.Vphi*kappa[i,:,np.newaxis]).flatten()
            V = np.concatenate((Vfull[self.idx_in], np.ones(self.nbdy)))
            
            A = coo_matrix((V, (self.Ibdy, self.Jbdy)), shape = (nvert, nvert)).tocsr()
            p[i, :] = spsolve(A, b)
            
        return p
    
#    @profile    
    def lnprob(self, u, observation):
        ob_idx = observation[0,:].astype(int)
        ob_data = observation[1,:]

        if u.ndim is 1:
            u = u[np.newaxis,:]
        batchsize = u.shape[0]
        u = np.reshape(u, (batchsize, -1))
        kappa = np.exp(u)
        
        nvert = self.n**2
        nface = (self.n-1)**2
        
        p = np.zeros((batchsize, nvert), dtype=float)
        F = np.zeros(batchsize, dtype=float)
        duF = np.zeros((batchsize, nface), dtype=float)
        
        b = self.b.copy()
        b[self.idx] = 0

        if not self.lnprob_likelihood:
            inv_Gamma_x_u = self.inv_Gamma.dot(u.T)
        
        for i in range(batchsize):
            Vfull = (self.Vphi*kappa[i,:,np.newaxis]).flatten()
            V = np.concatenate((Vfull[self.idx_in], np.ones(self.nbdy)))# dA/du in sparse form
            
            A = coo_matrix((V, (self.Ibdy, self.Jbdy)), shape = (nvert, nvert)).tocsr()
            p[i, :] = spsolve(A, b)
            
            # Solve the adjoint gradient
#            Vpart = np.zeros_like(Vfull)
#            Vpart[self.idx_in] = Vfull[self.idx_in]
            duG = coo_matrix((p[i, self.J]*Vfull, (np.arange(0, 16*nface, 1)//16, self.I)), shape = ((nface, nvert))).tocsr()
            duG[:, self.idx] = np.zeros((nface,1))
            
            # Log of Prob, and its Gradient
            r = np.zeros(shape=(nvert))
            r[ob_idx] = ob_data-p[i, ob_idx]
            if self.lnprob_likelihood:
                F[i] = -0.5/self.sigma**2*np.linalg.norm(r)**2
                duF[i, :] = -1/(self.sigma**2)*duG.dot(spsolve(A.transpose(), r))
            else:
                F[i] = -0.5*u[i,:].dot(inv_Gamma_x_u[:, i])-0.5/self.sigma**2*np.linalg.norm(r)**2
                duF[i, :] = -(inv_Gamma_x_u[:, i]+1/(self.sigma**2)*duG.dot(spsolve(A.transpose(), r)))
        return (p, F, duF)
  
    # Generate 2D quadrilateral mesh on [0,1]x[0,1]
    def meshgen_2dsq(self):
        y, x = np.meshgrid(np.linspace(0,1,self.n),np.linspace(0,1,self.n))
        vertex = np.column_stack((x.reshape(-1),y.reshape(-1)))
        
        index1 = np.asarray(np.where((np.linspace(1,(self.n)*(self.n-1),(self.n)*(self.n-1)) % self.n) != 0)).flatten()
        index3 = np.asarray(np.where((np.linspace(self.n+1,self.n**2,self.n**2-self.n) % self.n) != 0)).flatten() + self.n
        face = np.column_stack((index1,index1+1,index3+1,index3))
        
        bdyy0 = np.column_stack((np.arange(0,self.n*(self.n-2)+1,self.n),np.arange(self.n,self.n*(self.n-1)+1,self.n)))
        bdyx1 = np.column_stack((np.arange(self.n*(self.n-1),self.n**2-1,1),np.arange(self.n*(self.n-1)+1,self.n**2,1)))
        bdyy1 = np.column_stack((np.arange(self.n**2-1,2*self.n-2,-self.n),np.arange(self.n*(self.n-1)-1,self.n-2,-self.n)))
        bdyx0 = np.column_stack((np.arange(self.n-1,0,-1),np.arange(self.n-2,-1,-1)))
        bdy = np.row_stack((bdyy0,bdyx1,bdyy1,bdyx0))
        
        return (face, vertex, bdy)
    
    def hessian_gnapprox(self, u, observation):
        # Gaussian-Newton approximation of the Hessian
        ob_idx = observation[0,:].astype(int)
        m = ob_idx.size

        u = np.reshape(u, -1)
        kappa = np.exp(u)
        
        nvert = self.n**2
        nface = (self.n-1)**2
        
        M = np.zeros((m,nvert), dtype=float)
        M[np.arange(m, dtype=int),ob_idx] = np.ones(m)
        p = np.zeros(nvert, dtype=float)
        H = np.zeros((nface, nface), dtype=float)
        
        Vfull = (self.Vphi*kappa[:,np.newaxis]).flatten()
        V = np.concatenate((Vfull[self.idx_in], np.ones(self.nbdy)))# dA/du in sparse form
            
        b = self.b.copy()
        b[-1] = 0
        
        A = coo_matrix((V, (self.Ibdy, self.Jbdy)), shape = (nvert, nvert)).tocsr()
        p = spsolve(A, b)

        # Solve the adjoint gradient        
        duG = coo_matrix( (p[self.J]*Vfull, (np.arange(0, 16*nface, 1)//16, self.I)), shape = ((nface, nvert)))
        
        J = 1/self.sigma*duG.dot(spsolve(A.transpose(), M.transpose()))
        H = J@J.transpose()
            
        return H