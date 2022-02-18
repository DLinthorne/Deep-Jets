# Generic/Built-in
import os
import sys

# Other Libs
from scipy.interpolate import UnivariateSpline
from scipy.sparse import dok_matrix
import numpy as np
import pickle

def get_pdg_code(name):
    switcher = {
        "e+": -11,
        "e-": 11,
        "mu+": -13,
        "mu-": 13,
        "met": 12,
        "ga": 22,
        "pi+": 211,
        "pi-": -211
    }
    return switcher.get(name, "nothing")

def sub_pdg_code(name):
    switcher = {
        -11: "e+",
        11: "e-",
        -13: "mu+",
        13: "mu-",
        12: "met",
        22: "ga",
        211: "pi+",
        -211: "pi-"
    }
    return switcher.get(name, "nothing")

def r_z_to_eta(r, z):
    """
    Returns:
    eta -- object's pesudorapidity relative to beam axis
    """
    if r != 0.:
        return np.arcsinh(z / r)
    else:
        return float('Inf')

def eta_r_to_z(eta, r):
    """
    Returns:
    z   -- object's longitudinal coordinate aligned with beam axis
    """
    return r * np.sinh(eta)


def eta_z_to_r(eta, z):
    """
    Returns:
    r -- radial distance from the beamline
    """
    if eta != 0.:
        return z / np.sinh(eta)
    elif z == 0.:
        return 0.
    else:
        return float('Inf')

def rcycl(ptphietam, q, B):

    InTeslaOverGeV2 = 1.69041E+16
    InGeVcm = 1.97327E-14
    UnitsRS = (InTeslaOverGeV2 * InGeVcm)

    tmp = 0.0
    try:
        tmp = ptphietam[0] / (B * abs(q)) * UnitsRS
    except:
        return float('Inf')
    else:
        return tmp


def px(ptphietam):
    return ptphietam[0] * np.cos(ptphietam[1])

def py(ptphietam):
    return ptphietam[0] * np.sin(ptphietam[1])

def pz(ptphietam):
    return ptphietam[0] * np.sinh(ptphietam[2])

def pmod(ptphietam):
    return ptphietam[0] * np.cosh(ptphietam[2])

def energy(ptphietam):
    return np.sqrt(pmod(ptphietam) ** 2. + ptphietam[3] ** 2)

def pt_four_vec(px, py, pz, e, m):
    mod = np.sqrt(px**2. + py**2.)
    if mod != 0.:
        return [mod, np.arctan2(py, px), np.arctanh(pz/np.sqrt(mod**2.+ pz**2.)), m]
    else:
        return [0., 0., float('Inf')*np.sign(pz), m]

def to_cartesian(ptphietam):
    return [ptphietam[0]*np.cos(ptphietam[1]), ptphietam[0]*np.sin(ptphietam[1]),
            ptphietam[0]*np.sinh(ptphietam[2]), np.sqrt((ptphietam[0]*np.cosh(ptphietam[2]))**2
                                                        + ptphietam[3]), ptphietam[3]]


def OmegaCycl(ptphietam, q, B):
    InTeslaOverGeV2 = 1.69041E+16
    InGeVcm = 1.97327E-14
    IncmOverns = 29.9792
    UnitsOmegaS = (IncmOverns / (InTeslaOverGeV2 * InGeVcm))

    tmp = 0.0
    try:
        tmp = (B * abs(q)) / energy(ptphietam) * UnitsOmegaS
    except:
        return float('Inf')
    else:
        return tmp


def can_reach_r(nextR, rcenter, rcycl):
    return (nextR >= abs(rcenter - rcycl)) and (nextR <= rcenter + rcycl)


def next_time_from_r(nextR, t0, omega, q, rcenter, phicenter, rcycl, phicycl):
    delta = 0.
    if can_reach_r(nextR, rcenter, rcycl):
        delta = np.arccos((nextR**2. - rcenter**2. - rcycl**2.) /
                       (2.*rcenter*rcycl))
    else:
        return float('Inf')
    delta_tp = 1./(np.sign(q)*omega)*(fmod(delta - phicycl + phicenter, 2.*np.pi))
    delta_tm = 1./(np.sign(q)*omega)*(fmod(-delta - phicycl + phicenter, 2.*np.pi))
    delta_period = 1./(np.sign(q)*omega)*2*np.pi
    return t0 + min([x for x in ([delta_tp + i*delta_period for i in range(-1, 2)]
                                 + [delta_tm + i*delta_period for i in range(-1, 2)]) if x > (1.e-10/omega)])


def next_time_from_z(next_z, t0, z_center, ptphietam):
    IncmOverns = 29.9792
    UnitsTime = IncmOverns

    tmp = 0.
    try:
        tmp = ((next_z - z_center) * energy(ptphietam) / (pz(ptphietam) * UnitsTime))
        if tmp >= 0.:
            return tmp + t0
        else:
            return float('Inf')
    except:
        return float('Inf')


def z_at_time(ptphietam, t, t0, z_center):
    IncmOverns = 29.9792
    UnitsTime = IncmOverns

    if np.isinf(t):
        return float('Inf')
    else:
        try:
            return (pz(ptphietam) / energy(ptphietam)) * (t - t0) * UnitsTime + z_center
        except:
            return z_center


def r_at_time(t, t0, omega, q, r_center, phi_center, rcycl, phicycl):
    if np.isinf(t) or np.isinf(t0):
        return float('Inf')
    elif t == t0:
        return np.sqrt(r_center ** 2. + rcycl ** 2.
                       + 2. * rcycl * r_center * np.cos(phicycl - phi_center))
    else:
        try:
            return np.sqrt(r_center ** 2. + rcycl ** 2.
                           + 2. * rcycl * r_center * np.cos(np.sign(q) * omega * (t - t0)
                           + phicycl - phi_center))
        except:
            return float('Inf')


def phi_at_time(t, t0, omega, q, r_center, phi_center, rcycl, phicycl):
    if np.isinf(t) or np.isinf(t0):
        return float('Inf')
    elif t == t0:
        return np.arctan2(r_center * np.sin(phi_center) +
                          rcycl * np.sin(phicycl),
                          r_center * np.cos(phi_center) +
                          rcycl * np.cos(phicycl))
    else:
        return np.arctan2(r_center * np.sin(phi_center) +
                          rcycl * np.sin(np.sign(q) * omega * (t - t0) + phicycl),
                          r_center * np.cos(phi_center) +
                          rcycl * np.cos(np.sign(q) * omega * (t - t0) + phicycl))


def next_omega_cycl(old_omega_cycl, frac_loss, delta_rel_loss):
    if np.exp(-frac_loss) - delta_rel_loss > 0.:
        return old_omega_cycl / (np.exp(-frac_loss) - delta_rel_loss)
    else:
        return float('Inf')


def next_phi_cycl(newt, old_t, omega, q, old_phicycl):
    return np.sign(q) * omega * (newt - old_t) + old_phicycl


def next_ptphietam(ptphietam, newomegacycl, oldomegacycl, q, newphicycl):
    if newomegacycl == oldomegacycl:
        return [ptphietam[0], fmod(newphicycl + np.pi / 2. * np.sign(q), 2*np.pi),
                ptphietam[2], ptphietam[3]]
    else:
        if(not np.isinf(newomegacycl) and (1. - ((newomegacycl / oldomegacycl)**2. - 1.)
                                           * (ptphietam[3] / (pmod(ptphietam))) ** 2.) > 0):
            return [ptphietam[0] * oldomegacycl / newomegacycl
                    * np.sqrt(1. - ((newomegacycl / oldomegacycl)**2. - 1.)
                    * (ptphietam[3] / (pmod(ptphietam))) ** 2.),
                    fmod(newphicycl + np.pi / 2. * np.sign(q),2*np.pi), ptphietam[2], ptphietam[3]]
        else:
            return [0., fmod(newphicycl + np.pi / 2. * np.sign(q),2*np.pi), ptphietam[2], ptphietam[3]]


def next_r_cycl(old_rcycl, new_ptphietam, old_ptphietam):
    return new_ptphietam[0] / old_ptphietam[0] * old_rcycl


def next_r_center(new_rcycl, new_phicycl, old_r_center, old_phi_center, old_rcycl):
    return np.sqrt(old_r_center ** 2. + (old_rcycl - new_rcycl) ** 2. + 2. * old_r_center *
                (old_rcycl - new_rcycl) * np.cos(old_phi_center - new_phicycl))


def next_phi_center(new_rcycl, new_phicycl, old_r_center, old_phi_center, old_rcycl):
    return np.arctan2(old_r_center * np.sin(old_phi_center) + (old_rcycl - new_rcycl) * np.sin(new_phicycl),
                      old_r_center * np.cos(old_phi_center) + (old_rcycl - new_rcycl) * np.cos(new_phicycl))

def adjust_origin(line, pdg, z_origin, rcalo, zcalo):
    if abs(pdg) == 13:
        return line
    else:
        pmod = line[0]*np.cosh(line[2])
        new_eta = 0.
        if abs(line[2]) > r_z_to_eta(rcalo, zcalo):
            new_eta = r_z_to_eta(eta_z_to_r(abs(line[2]), zcalo), zcalo * np.sign(line[2]) - z_origin)
        else:
            new_eta = r_z_to_eta(rcalo, eta_z_to_r(line[2], rcalo) - z_origin)
        return [pmod/np.cosh(new_eta), line[1], new_eta, line[3]]


def add_photons(calo_depo, calo_size, eta_range, barcode_max):
    gamma_list = [[(i[0] + 0.5) * calo_size[0], (i[1] + 0.5) * calo_size[1] + eta_range[0],
                  calo_depo[i[0], i[1]]] for i in np.asarray(calo_depo.nonzero()).transpose()]

    gamma_list_2 = [[x[0], x[1], x[2]] if x[1] <= eta_range[1] else
                  [x[0], x[1] - 2. * eta_range[1], x[2]] for x in gamma_list]
    if len(gamma_list_2) > 0:
        return [['P', barcode_max+i+1, 22] + to_cartesian([x[2] / np.cosh(x[1]), x[0], x[1], 0.0]) +
                [1, 0., 0., 0, 0] for i, x in enumerate(gamma_list_2)], barcode_max + len(gamma_list_2)
    else:
        return [], barcode_max


def read_grid(fname, with_header=True):
    """
    Arguments:
    fname  -- file path to grids
    with_header -- flag on whether grid files have header data

    Returns:
    func   -- list of interpolated splines of layer grids
    ranges -- list of layer dimensions in (r,z,?)
    """
    ranges = []
    header_flag = 0
    if with_header:
        with open(fname, 'r') as f:
            ranges = [float(x) for x in f.readline().split(',')]
        header_flag = 1

    vals = np.genfromtxt(fname, delimiter=',', skip_header=header_flag)
    funcs = [UnivariateSpline(vals[:, 0], vals[:, i], k=1) for i in range(1, len(vals[0]))]
    [elem.set_smoothing_factor(0.) for elem in funcs]

    return [funcs, ranges]


def init_layers_atlas(atlas_dir ="/Users/dlinthorne/Projects/MISC/IDsym_Daniel_&_Dylan/PythonSym/"):
    """
    Initializes the tracking layers using ATLAS geometric specifications:
    PIPE -> IBL -> PIXEL -> SCT -> TRT

    Arguments:
    ATLASdir -- path to ATLAS geometry files

    Returns:
    layers_e -- interpolated slines of each layer relevant for electron propagation
    layers_mu -- interpolated slines of each layer relevant for muon propagation
    layers_pi -- interpolated slines of each layer relevant for pion propagation
    """

    # test git
    pipe = read_grid(atlas_dir + 'ATLASEMlossPipe.csv')
    mat = read_grid(atlas_dir + 'ATLASEMlossService.csv')
    ibl = read_grid(atlas_dir + 'ATLASEMlossIBL.csv')
    pixel_barrel = [read_grid(atlas_dir + 'ATLASEMlossPixelB' + str(i) + '.csv') for i in range(1, 4)]
    pixel_wheel = [read_grid(atlas_dir + 'ATLASEMlossPixelW' + str(i) + '.csv') for i in range(1, 4)]
    sct_barrel = [read_grid(atlas_dir + 'ATLASEMlossSctB' + str(i) + '.csv') for i in range(1, 5)]
    sct_wheel = [read_grid(atlas_dir + 'ATLASEMlossSctW' + str(i) + '.csv') for i in range(1, 10)]
    trt_barrel = [read_grid(atlas_dir + 'ATLASEMlossTRTB' + str(i) + '.csv') for i in range(1, 5)]
    trt_wheel = [read_grid(atlas_dir + 'ATLASEMlossTRTW' + str(i) + '.csv') for i in range(1, 5)]

    dummy_loss = lambda x: 0.

    layer_positions = ([pipe[1]] + [ibl[1]] + [pixel_barrel[i][1] for i in range(0, 3)]
                       + [pixel_wheel[i][1] for i in range(0, 3)] + [mat[1]]
                       + [sct_barrel[i][1] for i in range(0, 4)]
                       + [sct_wheel[i][1] for i in range(0, 9)]
                       + [trt_barrel[i][1] for i in range(0, 4)]
                       + [trt_wheel[i][1] for i in range(0, 4)]
                       + [[115., 0, eta_r_to_z(1.5, 115.)],
                          [eta_z_to_r(3.5, eta_r_to_z(1.5, 115.)), 115., eta_r_to_z(1.5, 115.)]])

    layer_em_loss = ([pipe[0][0]] + [ibl[0][0]] + [pixel_barrel[i][0][0] for i in range(0, 3)]
                    + [pixel_wheel[i][0][0] for i in range(0, 3)] + [mat[0][0]]
                    + [sct_barrel[i][0][0] for i in range(0, 4)]
                    + [sct_wheel[i][0][0] for i in range(0, 9)]
                    + [trt_barrel[i][0][0] for i in range(0, 4)]
                    + [trt_wheel[i][0][0] for i in range(0, 4)]
                    + [dummy_loss, dummy_loss])

    layer_dummy_loss = [dummy_loss for i in range(0, len(layer_positions))]

    #todo: to be changed once grids are available

    layer_had_loss = layer_dummy_loss

    layer_dimensions = ([[None, None, None]] + [[0.005 / ibl[1][0], 0.025, 0.023]]
                        + [[0.005 / pixel_barrel[i][1][0], 0.04, 0.025] for i in range(0, 3)]
                        + [[0.005 / pixel_wheel[i][1][0], 0.04, 0.025] for i in range(0, 3)]
                        + [[None, None, None]]
                        + [[0.008 / sct_barrel[i][1][0], 6.39, 0.0285] for i in range(0, 4)]
                        + [[0.006 / sct_wheel[i][1][0], 6.39, 0.0285] for i in range(0, 9)]
                        + [[None, None, None] for i in range(0, 4)]
                        + [[None, None, None] for i in range(0, 4)]
                        + [[2. * np.pi / 256., 0.025, None], [2. * np.pi / 256., 0.025, None]])

    layer_orient = ([True]
                    + [True]
                    + [True, True, True]
                    + [False, False, False]
                    + [True]
                    + [True, True, True, True]
                    + [False, False, False, False]
                    + [False, False, False, False, False]
                    + [True, True, True, True]
                    + [False, False, False, False]
                    + [True, False])

    layers_e = [[a, b, c, d] for a, b, c, d in zip(layer_positions, layer_em_loss, layer_dimensions, layer_orient)]
    layers_mu = [[a, b, c, d] for a, b, c, d in zip(layer_positions, layer_dummy_loss, layer_dimensions, layer_orient)]
    layers_pi = [[a, b, c, d] for a, b, c, d in zip(layer_positions, layer_had_loss, layer_dimensions, layer_orient)]

    return (layers_e, layers_mu, layers_pi)


def init_ionization(atlas_dir ="/Users/dlinthorne/Projects/MISC/IDsym_Daniel_&_Dylan/PythonSym/"):

    return (read_grid(atlas_dir + 'IonElossEles.csv', False)[0],
            read_grid(atlas_dir + 'IonElossMus.csv', False)[0],
            read_grid(atlas_dir + 'IonElossPis.csv', False)[0])


def process_event(hep_file, event_header, ecal_file, trk_file_list, layers_e, layers_mu,
                  layers_meson, ion_e, ion_mu, ion_meson, b_field=2):

    timeouts = [0]*len(layers_e)

    layer_hits = [dok_matrix((int(np.ceil(2. * np.pi / x[2][0])),
                              int(2. * np.ceil((x[0][2] - x[0][1]) / x[2][1]))))
                  if x[2][2] is not None and x[3] else
                  dok_matrix((int(np.ceil(2. * np.pi / x[2][0])),
                              int(2. * np.ceil((x[0][1] - x[0][0]) / x[2][1]))))
                  if x[2][2] is not None else None for x in layers_e]

    calo_photons = [dok_matrix((int(np.ceil(2. * np.pi / layers_e[-2][2][0])),
                                int(np.ceil(2. * r_z_to_eta(layers_e[-2][0][0], layers_e[-2][0][2]) /
                                            layers_e[-2][2][1])))),
                    dok_matrix((int(np.ceil(2. * np.pi / layers_e[-1][2][0])),
                                int(np.ceil(2. * (r_z_to_eta(layers_e[-1][0][0], layers_e[-1][0][2])
                                                  - r_z_to_eta(layers_e[-1][0][1], layers_e[-1][0][2])) /
                                            layers_e[-1][2][1]))))]

    num_vertices = int(event_header[8])
    # print numvertices
    retrieve_line = hep_file.readline()
    while   retrieve_line[0] !=  86:
    #    ecal_file.write(line)
        retrieve_line = hep_file.readline()
    z_origin = None
    offset = 0
    four_momentum = [0., 0., 0., 0.]
    max_barcode = 0
    for vertex_index in range(num_vertices):

        # now got the first vertex, extract zorigin
        vertex = retrieve_line.split()

        if z_origin is None:
            z_origin = float(vertex[5])/10
        # print "zorigin (input):"+str(zorigin)+"cm"

        #read vertex by vertex
        num_in = int(vertex[7])
        num_out = int(vertex[8])
        perp_vert_pos = [float(vertex[3]), float(vertex[4])]
        vert_pos = [np.sqrt(perp_vert_pos[0]**2 + perp_vert_pos[1]**2)/10,
                    np.arctan2(perp_vert_pos[1], perp_vert_pos[0]), float(vertex[5])/10]

        particles = [hep_file.readline().split() for x in range(num_in + num_out)]
        #print(particles)
        max_barcode = max([max_barcode]+[int(x[1]) for x in particles])

        ready_particles = [x for x in particles if x[8] != '1' or abs(int(x[2])) not in [11, 13, 211, 321]
                           or abs(np.arctanh(float(x[5])/np.sqrt(float(x[3])**2 + float(x[4])**2 + float(x[5])**2))) > 3.5]

        particle_list = [x for x in particles if x[8] == '1' and
                         abs(int(x[2])) in [11, 13, 211, 321] and
                         abs(np.arctanh(float(x[5])/np.sqrt(float(x[3])**2 + float(x[4])**2 + float(x[5])**2))) <= 3.5]

        if len(particle_list) > 0:
            print(particle_list)
            #print"+++", [sum(i) for i in zip(*[[float(y) for y in x[3:7]]
            #                                     for x in particle_list])]
            four_momentum = [(a+b) for a, b in
                          zip(four_momentum, [sum(i) for i in
                                           zip(*[[float(y) for y in x[3:7]]
                                                 for x in particle_list])])]
        calo_particles = []
        tmp_offset = 0
        for particle in particle_list:
            pdg = int(particle[2])
            barcode = int(particle[1])
            ptphietam = pt_four_vec(float(particle[3]), float(particle[4]),
                                    float(particle[5]), float(particle[6]),
                                    float(particle[7]))
            tmphits = None
            tmpphos = None
            calopart = None
            tmptimeouts = None
            if abs(pdg) == 11:
                #print ptphietam
                calopart, tmphits, tmpphos, tmptimeouts = \
                    Propagate(ptphietam, -np.sign(pdg), b_field, vert_pos,
                              ion_e, layers_e, 0., True, True)
            elif abs(pdg) == 13:
                #print ptphietam
                calopart, tmphits, tmpphos, tmptimeouts = \
                    Propagate(ptphietam, -np.sign(pdg), b_field, vert_pos,
                              ion_mu, layers_mu, 0., False, False)
            elif abs(pdg) > 100:
                #print ptphietam
                calopart, tmphits, tmpphos, tmptimeouts = \
                    Propagate(ptphietam, -np.sign(pdg), b_field, vert_pos,
                              ion_meson, layers_meson, 0., False, True)
            if calopart is not None:
                #print calopart
                calo_particles.append(particle[:3] +
                                      to_cartesian(adjust_origin(calopart, pdg, z_origin,
                                                                 layers_e[-2][0][0],
                                                                 layers_e[-1][0][2])) + particle[8:])
            else:
                tmp_offset -= 1
            if tmpphos is not None:
                calo_photons[0] = calo_photons[0] + tmpphos[0]
                calo_photons[1] = calo_photons[1] + tmpphos[1]
            if tmphits is not None:
                layer_hits = [(f + t) if (f is not None and t is not None)
                             else None for f, t in zip(layer_hits, tmphits)]
            if tmptimeouts is not None:
                timeouts = [(f + t) for f, t in zip(timeouts, tmptimeouts)]
        #last vertex: add all the brem photons and met
        #print caloparticles
        if vertex_index == num_vertices - 1:
            list_parts, max_barcode = add_photons(calo_photons[0], layers_e[-2][2],
                                                  [0, r_z_to_eta(layers_e[-2][0][0],
                                                                layers_e[-2][0][2])],
                                                  max_barcode)
            calo_particles.extend(list_parts)
            list_parts, max_barcode = add_photons(calo_photons[1], layers_e[-1][2],
                                                  [r_z_to_eta(layers_e[-1][0][1],
                                                             layers_e[-1][0][2]),
                                                  r_z_to_eta(layers_e[-1][0][0],
                                                             layers_e[-1][0][2])],
                                                  max_barcode)
            calo_particles.extend(list_parts)
        if len(calo_particles) > 0:
            # print "--", [sum(i) for i in zip(*[[float(y) for y in x[3:7]] for x
            #                                    in caloparticles])]
            four_momentum = [(a-b) for a, b in
                          zip(four_momentum, [sum(i) for i
                                           in zip(*[[float(y) for y in x[3:7]] for x
                                                    in calo_particles])])]
        if vertex_index == num_vertices - 1:
            pmod = np.sqrt(four_momentum[0]**2 + four_momentum[1]**2 + four_momentum[2]**2)
            if four_momentum[3] < pmod:
                four_momentum[3] = pmod
            calo_particles.append(['P', str(max_barcode + 1), 12] + four_momentum
                                  + [np.sqrt(four_momentum[3]**2 - pmod**2.), 1, 0., 0., 0, 0])

        out_particles = [' '.join([str(y) for y in x]) for x in ready_particles + calo_particles]
        print(out_particles)
        #ecal_file.write(' '.join(vertex[:5] +
        #                         [str((vert_pos[2]-z_origin)*10)] + vertex[6:8] +
        #                         [str(len(out_particles)-num_in)] + vertex[9:]) + '\n')
        #[ecal_file.write(x + '\n') for x in out_particles]
        #offset += tmp_offset
        #read next vertex
        retrieve_line = hep_file.readline()
    #write hits (only IBL)
    write_layer_events(layer_hits[1], timeouts[1] / max(len(particle_list), 1), event_header, trk_file_list[0])
    #write hits (all layers)
    #[WriteLayersEvent(layerHits[idx], timeouts[idx]/max(len(particle_list),1),event_header,trackerfilelist[idx-1])
    # for idx in range(1,5)]
    #WriteLayersEvent(layer, idx, timeouts[idx]/len(particle_list), trackfile)
    # for idx, layer in enumerate(layerHits) if layer is not None]
    #return next event header
    return retrieve_line


def write_layer_events(layer, timeout_frac, event_header, track_file):

    print("test")
    print(timeout_frac)
    print(layer)
    #pickle.dump(str(event_header[1]) + " " + str(timeout_frac), track_file, protocol=0)

    #with open(track_file, 'wb') as pickle_file:
    #    pickle.dump(layer, pickle_file)
    #pickle.dump(layer, track_file, protocol=0)


if __name__ == '__main__':

    data_dir = "/Users/dlinthorne/Projects/MISC/IDsym_Daniel_&_Dylan/"

    atlas_dir = data_dir + "PythonSym/"
    hep_file = data_dir + "truth/Signal_truth.hepmc"
    tracker_file_temp = data_dir + "tracker/" + "test_truth.hepmc".replace("truth", "tracker_LAYERID").replace(".hepmc",
                                                                                                             ".txt")
    tracker_file_list = [tracker_file_temp.replace("LAYERID", str(idx-1)) for idx in range(1, 5)]

    os.system("rm -f " + hep_file.replace("truth", "Ecal"))
    Ecal_file = open(hep_file.replace("truth", "Ecal"), "wb")

    atlas_bfield = 2

    layers_atlas_e, layers_atlas_mu, layers_atlas_pi = init_layers_atlas(atlas_dir)
    ion_e, ion_mu, ion_pi = init_ionization(atlas_dir)

   # with open(hep_file, 'rb') as input_file:
    #    for index, line_string in enumerate(input_file):

    #        if line_string[0] in [85, 72]:
    #            continue
    #        if line_string[0] == 69:
    #            print(line_string.split())
    #            event_header = line_string.split()
    #            line = process_event(input_file, event_header, Ecal_file, tracker_file_list,
    #                                 layers_atlas_e, layers_atlas_mu, layers_atlas_pi, ion_e,
    #                                 ion_mu, ion_pi, atlas_bfield)
    input_file = open(hep_file, 'rb')

    line = input_file.readline()

    while line[0] != 69:
        #print(line[0])
        line = input_file.readline()
    while line[0] == 69:
        #print(line.split())
        event_header = line.split()
        line = process_event(input_file, event_header, Ecal_file, tracker_file_list,
                             layers_atlas_e, layers_atlas_mu, layers_atlas_pi, ion_e,
                             ion_mu, ion_pi, atlas_bfield)

    Ecal_file.close()
    input_file.close()

    print('end')
