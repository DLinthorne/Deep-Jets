# Generic/Built-in
import os

# Other Libs
from scipy.interpolate import UnivariateSpline
import numpy as np
import pickle


def eta_r_to_z(eta, r):
    """
    Arguments:
    eta -- object's pesudorapidity relative to beam axis
    r -- radial distance from the beamline

    Returns:
    z   -- object's longitudinal coordinate aligned with beam axis
    """
    return r * np.sinh(eta)


def eta_z_to_r(eta, z):
    """
    Arguments:
    eta -- object's pesudorapidity relative to beam axis
    z   -- object's longitudinal coordinate aligned with beam axis

    Returns:
    r -- radial distance from the beamline
    """
    if eta != 0.:
        return z / np.sinh(eta)
    elif z == 0.:
        return 0.
    else:
        return float('Inf')


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


def init_layers_atlas(ATLASdir ="/Users/dlinthorne/Projects/MISC/IDsym_Daniel_&_Dylan/PythonSym/"):
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
    pipe = read_grid(ATLASdir + 'ATLASEMlossPipe.csv')
    mat = read_grid(ATLASdir + 'ATLASEMlossService.csv')
    IBL = read_grid(ATLASdir + 'ATLASEMlossIBL.csv')
    pixel_barrel = [read_grid(ATLASdir + 'ATLASEMlossPixelB' + str(i) + '.csv') for i in range(1, 4)]
    pixel_wheel = [read_grid(ATLASdir + 'ATLASEMlossPixelW' + str(i) + '.csv') for i in range(1, 4)]
    sct_barrel = [read_grid(ATLASdir + 'ATLASEMlossSctB' + str(i) + '.csv') for i in range(1, 5)]
    sct_wheel = [read_grid(ATLASdir + 'ATLASEMlossSctW' + str(i) + '.csv') for i in range(1, 10)]
    trt_barrel = [read_grid(ATLASdir + 'ATLASEMlossTRTB' + str(i) + '.csv') for i in range(1, 5)]
    trt_wheel = [read_grid(ATLASdir + 'ATLASEMlossTRTW' + str(i) + '.csv') for i in range(1, 5)]

    dummy_loss = lambda x: 0.

    layer_positions = ([pipe[1]] + [IBL[1]] + [pixel_barrel[i][1] for i in range(0, 3)]
                       + [pixel_wheel[i][1] for i in range(0, 3)] + [mat[1]]
                       + [sct_barrel[i][1] for i in range(0, 4)]
                       + [sct_wheel[i][1] for i in range(0, 9)]
                       + [trt_barrel[i][1] for i in range(0, 4)]
                       + [trt_wheel[i][1] for i in range(0, 4)]
                       + [[115., 0, eta_r_to_z(1.5, 115.)],
                          [eta_z_to_r(3.5, eta_r_to_z(1.5, 115.)), 115., eta_r_to_z(1.5, 115.)]])

    layer_EMloss = ([pipe[0][0]] + [IBL[0][0]] + [pixel_barrel[i][0][0] for i in range(0, 3)]
                    + [pixel_wheel[i][0][0] for i in range(0, 3)] + [mat[0][0]]
                    + [sct_barrel[i][0][0] for i in range(0, 4)]
                    + [sct_wheel[i][0][0] for i in range(0, 9)]
                    + [trt_barrel[i][0][0] for i in range(0, 4)]
                    + [trt_wheel[i][0][0] for i in range(0, 4)]
                    + [dummy_loss, dummy_loss])

    layer_dummy_loss = [dummy_loss for i in range(0, len(layer_positions))]

    #todo: to be changed once grids are available

    layer_HADloss = layer_dummy_loss
    layer_dimensions = ([[None, None, None]] + [[0.005 / IBL[1][0], 0.025, 0.023]]
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

    layers_e = [[a, b, c, d] for a, b, c, d in zip(layer_positions, layer_EMloss, layer_dimensions, layer_orient)]
    layers_mu = [[a, b, c, d] for a, b, c, d in zip(layer_positions, layer_dummy_loss, layer_dimensions, layer_orient)]
    layers_pi = [[a, b, c, d] for a, b, c, d in zip(layer_positions, layer_HADloss, layer_dimensions, layer_orient)]

    return (layers_e, layers_mu, layers_pi)


def init_ionization(ATLASdir ="/Users/dlinthorne/Projects/MISC/IDsym_Daniel_&_Dylan/PythonSym/"):

    return (read_grid(ATLASdir + 'IonElossEles.csv', False)[0],
            read_grid(ATLASdir + 'IonElossMus.csv', False)[0],
            read_grid(ATLASdir + 'IonElossPis.csv', False)[0])


def write_layer_events(layer, timeoutfrac, eventheader, trackfile):

    pickle.dump(eventheader[1]+" "+str(timeoutfrac), trackfile,  protocol=0)
    pickle.dump(layer, trackfile,  protocol=0)


def process_event(hepmcfile, event_header, ecalfile, trackerfilelist, layers_e, layers_mu,
                  layers_meson, ion_e, ion_mu, ion_meson, Bfield=2):

    timeouts = [0]*len(layers_e)

    layerHits = [dok_matrix((int(np.ceil(2. * np.pi / x[2][0])),
                             int(2. * np.ceil((x[0][2] - x[0][1]) / x[2][1]))))
                 if x[2][2] is not None and x[3] else
                 dok_matrix((int(np.ceil(2. * np.pi / x[2][0])),
                             int(2. * ceil((x[0][1] - x[0][0]) / x[2][1]))))
                 if x[2][2] is not None else None for x in layers_e]

    calophotons = [dok_matrix((int(ceil(2. * pi / layers_e[-2][2][0])),
                               int(ceil(2. * Eta(layers_e[-2][0][0],
                                                 layers_e[-2][0][2]) /
                                        layers_e[-2][2][1])))),
                   dok_matrix((int(ceil(2. * pi / layers_e[-1][2][0])),
                               int(ceil(2. * (Eta(layers_e[-1][0][0],
                                                  layers_e[-1][0][2]) -
                                              Eta(layers_e[-1][0][1],
                                                  layers_e[-1][0][2])) /
                                        layers_e[-1][2][1]))))]

    numvertices = int(event_header[8])
    # print numvertices
    line = hepmcfile.readline()
    while line[0] !=  86:
        ecalfile.write(line)
        line = hepmcfile.readline()
    zorigin = None
    offset = 0
    pxpypzEMET = [0., 0., 0., 0.]
    max_barcode = 0
    for nvertex in range(numvertices):
        #now got the first vertex, extract zorigin
        vertex = line.split()
        if zorigin is None:
            zorigin = float(vertex[5])/10
        # print "zorigin (input):"+str(zorigin)+"cm"
        #read vertex by vertex
        num_in = int(vertex[7])
        num_out = int(vertex[8])
        perpvpos = [float(vertex[3]), float(vertex[4])]
        vpos = [sqrt(perpvpos[0]**2 + perpvpos[1]**2)/10, arctan2(perpvpos[1], perpvpos[0]),
                float(vertex[5])/10]
        particles = [hepmcfile.readline().split() for x in range(num_in+num_out)]
        max_barcode = max([max_barcode]+[int(x[1]) for x in particles])
        readyparticles = [x for x in particles if x[8] != '1' or
                          abs(int(x[2])) not in [11, 13, 211, 321] or
                          abs(arctanh(float(x[5])/sqrt(float(x[3])**2+
                                                       float(x[4])**2+
                                                       float(x[5])**2))) > 3.5]
        particle_list = [x for x in particles if x[8] == '1' and
                         abs(int(x[2])) in [11, 13, 211, 321] and
                         abs(arctanh(float(x[5])/sqrt(float(x[3])**2+
                                                      float(x[4])**2+
                                                      float(x[5])**2))) <= 3.5]
        # print len(particle_list)
        if len(particle_list) > 0:
            # print "+++", [sum(i) for i in zip(*[[float(y) for y in x[3:7]]
            #                                     for x in particle_list])]
            pxpypzEMET = [(a+b) for a, b in
                          zip(pxpypzEMET, [sum(i) for i in
                                           zip(*[[float(y) for y in x[3:7]]
                                                 for x in particle_list])])]
        caloparticles = []
        tmpoffset = 0
        for particle in particle_list:
            pdg = int(particle[2])
            barcode = int(particle[1])
            ptphietam = ToPtPhiEtaM(float(particle[3]), float(particle[4]),
                                    float(particle[5]), float(particle[6]),
                                    float(particle[7]))
            tmphits = None
            tmpphos = None
            calopart = None
            tmptimeouts = None
            if abs(pdg) == 11:
                #print ptphietam
                calopart, tmphits, tmpphos, tmptimeouts = \
                    Propagate(ptphietam, -sign(pdg), Bfield, vpos,
                              ion_e, layers_e, 0., True, True)
            elif abs(pdg) == 13:
                #print ptphietam
                calopart, tmphits, tmpphos, tmptimeouts = \
                    Propagate(ptphietam, -sign(pdg), Bfield, vpos,
                              ion_mu, layers_mu, 0., False, False)
            elif abs(pdg) > 100:
                #print ptphietam
                calopart, tmphits, tmpphos, tmptimeouts = \
                    Propagate(ptphietam, -sign(pdg), Bfield, vpos,
                              ion_meson, layers_meson, 0., False, True)
            if calopart is not None:
                #print calopart
                caloparticles.append(particle[:3] +
                                     ToCartesian(AdjustOrigin(calopart, pdg, zorigin,
                                                              layers_e[-2][0][0],
                                                              layers_e[-1][0][2])) + particle[8:])
            else:
                tmpoffset -= 1
            if tmpphos is not None:
                calophotons[0] = calophotons[0] + tmpphos[0]
                calophotons[1] = calophotons[1] + tmpphos[1]
            if tmphits is not None:
                layerHits = [(f + t) if (f is not None and t is not None)
                             else None for f, t in zip(layerHits, tmphits)]
            if tmptimeouts is not None:
                timeouts = [(f + t) for f, t in zip(timeouts, tmptimeouts)]
        #last vertex: add all the brem photons and met
        #print caloparticles
        if nvertex == numvertices - 1:
            list_parts, max_barcode = AddPhotons(calophotons[0], layers_e[-2][2],
                                                 [0, Eta(layers_e[-2][0][0],
                                                         layers_e[-2][0][2])],
                                                 max_barcode)
            caloparticles.extend(list_parts)
            list_parts, max_barcode = AddPhotons(calophotons[1], layers_e[-1][2],
                                                 [Eta(layers_e[-1][0][1],
                                                      layers_e[-1][0][2]),
                                                  Eta(layers_e[-1][0][0],
                                                      layers_e[-1][0][2])],
                                                 max_barcode)
            caloparticles.extend(list_parts)
        if len(caloparticles) > 0:
            # print "--", [sum(i) for i in zip(*[[float(y) for y in x[3:7]] for x
            #                                    in caloparticles])]
            pxpypzEMET = [(a-b) for a, b in
                          zip(pxpypzEMET, [sum(i) for i
                                           in zip(*[[float(y) for y in x[3:7]] for x
                                                    in caloparticles])])]
        if nvertex == numvertices - 1:
            pmod = sqrt(pxpypzEMET[0]**2+pxpypzEMET[1]**2+pxpypzEMET[2]**2)
            if pxpypzEMET[3] < pmod:
                pxpypzEMET[3] = pmod
            caloparticles.append(['P', str(max_barcode+1), 12]+pxpypzEMET +
                                 [sqrt(pxpypzEMET[3]**2-pmod**2.),
                                  1, 0., 0., 0, 0])
        outparticles = [' '.join([str(y) for y in x]) for x in readyparticles+caloparticles]
        ecalfile.write(' '.join(vertex[:5] +
                                [str((vpos[2]-zorigin)*10)] + vertex[6:8] +
                                [str(len(outparticles)-num_in)] + vertex[9:]) + '\n')
        [ecalfile.write(x+'\n') for x in outparticles]
        offset += tmpoffset
        #read next vertex
        line = hepmcfile.readline()
    #write hits (only IBL)
    write_layer_events(layerHits[1], timeouts[1] / max(len(particle_list), 1), event_header, trackerfilelist[0])
    #write hits (all layers)
    #[WriteLayersEvent(layerHits[idx], timeouts[idx]/max(len(particle_list),1),event_header,trackerfilelist[idx-1])
    # for idx in range(1,5)]
    #WriteLayersEvent(layer, idx, timeouts[idx]/len(particle_list), trackfile)
    # for idx, layer in enumerate(layerHits) if layer is not None]
    #return next event header
    return line

if __name__ == '__main__':

    data_dir = "/Users/dlinthorne/Projects/MISC/IDsym_Daniel_&_Dylan/"

    ATLASdir = data_dir + "PythonSym/"
    hep_file = data_dir + "truth/Bkg_truth.hepmc"
    tracker_file_temp = data_dir + "tracker/" + "Bkg_truth.hepmc".replace("truth", "tracker_LAYERID").replace(".hepmc",
                                                                                                             ".txt")
    tracker_file_list = [tracker_file_temp.replace("LAYERID", str(idx-1)) for idx in range(1, 5)]

    os.system("rm -f " + hep_file.replace("truth", "Ecal"))
    Ecal_file = open(hep_file.replace("truth", "Ecal"), "wb")

    atlas_bfield = 2

    layers_atlas_e, layers_atlas_mu, layers_atlas_pi = init_layers_atlas(ATLASdir)
    ion_e, ion_mu, ion_pi = init_ionization(ATLASdir)

    event_header = str()
    with open(hep_file, 'rb') as input_file:
        for index, line_string in enumerate(input_file):

            if line_string[0] in [85,72]:
                continue
            else:
                print(line_string.split())

    Ecal_file.close()
    input_file.close()

    print('end')
