import odl
import numpy as np


## 640geo
class initialization():
    def __init__(self):
        self.param = {}
        # self.reso = 512 / 416 * 0.03
        self.reso = 0.4

        # image
        self.param['nx_h'] = 512
        self.param['ny_h'] = 512
        # self.param['nx_h'] = 410
        # self.param['ny_h'] = 410
        # self.param['sx'] = self.param['nx_h']*self.reso
        # self.param['sy'] = self.param['ny_h']*self.reso

        ## view
        self.param['startangle'] = 0
        self.param['endangle'] = 2 * np.pi

        # self.param['nProj'] = 640
        self.param['nProj'] = 650   

        ## detector
        # self.param['su'] = 2*np.sqrt(self.param['sx']**2+self.param['sy']**2)
        # self.param['nu_h'] = 641
        # self.param['dde'] = 1075*self.reso
        # self.param['dso'] = 1075*self.reso
        self.param['su'] = 512    #410
        self.param['nu_h'] = 1024    #1024
        self.param['dde'] = 500
        self.param['dso'] = 1000

        self.param['u_water'] = 0.192


def build_geometry(param):
    # reco_space_h = odl.uniform_discr(
    #     min_pt=[-param.param['sx'] / 2.0, -param.param['sy'] / 2.0],
    #     max_pt=[param.param['sx'] / 2.0, param.param['sy'] / 2.0], shape=[param.param['nx_h'], param.param['ny_h']],
    #     dtype='float32')
    reco_space_h = odl.uniform_discr(
        min_pt=[-param.param['su'] / 2.0, -param.param['su'] / 2.0],
        max_pt=[param.param['su'] / 2.0, param.param['su'] / 2.0], shape=[param.param['nx_h'], param.param['ny_h']],
        dtype='float32')

    angle_partition = odl.uniform_partition(param.param['startangle'], param.param['endangle'],
                                            param.param['nProj'])

    # detector_partition_h = odl.uniform_partition(-(param.param['su'] / 2.0), (param.param['su'] / 2.0),
    #                                              param.param['nu_h'])
    
    detector_partition_h = odl.uniform_partition(-384, 384,
                                                 param.param['nu_h'])  #1500*512/1000/2=384

    #G geometry_h = odl.tomo.FanBeamGeometry(angle_partition, detector_partition_h,
    #                                       src_radius=param.param['dso'],
    #                                       det_radius=param.param['dde'])
    geometry_h = odl.tomo.FanBeamGeometry(angle_partition, detector_partition_h,
                                          src_radius=param.param['dso'],
                                          det_radius=param.param['dde'])

    ray_trafo_hh = odl.tomo.RayTransform(reco_space_h, geometry_h, impl='astra_cuda')
    return ray_trafo_hh