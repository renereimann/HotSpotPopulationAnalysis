import numpy as np
from scipy.interpolate import UnivariateSpline

def get_splined(path, names=["x", "y"], use=None, delimiter=", ", comments="#", clip=True, s=0, k=1):
    data = np.genfromtxt(path, delimiter=delimiter, comments=comments, names=names)
    if use is None:
        use = names
    if clip:
        data[use[1]][data[use[1]] < 0.] = 0.
    spline = UnivariateSpline(data[use[0]], data[use[1]], s=s, k=k)
    return spline

N_spots_hist_threshold = {}
N_spots_hist_threshold[3.00] = get_splined("Check_Poisson_Distribution_Stefans_diss_Threshold_3.00.txt", names=["nSpots", "rel. counts"])
N_spots_hist_threshold[3.25] = get_splined("Check_Poisson_Distribution_Stefans_diss_Threshold_3.25.txt", names=["nSpots", "rel. counts"])
N_spots_hist_threshold[3.50] = get_splined("Check_Poisson_Distribution_Stefans_diss_Threshold_3.50.txt", names=["nSpots", "rel. counts"])
N_spots_hist_threshold[3.75] = get_splined("Check_Poisson_Distribution_Stefans_diss_Threshold_3.75.txt", names=["nSpots", "rel. counts"])
N_spots_hist_threshold[4.00] = get_splined("Check_Poisson_Distribution_Stefans_diss_Threshold_4.00.txt", names=["nSpots", "rel. counts"])
N_spots_hist_threshold[4.25] = get_splined("Check_Poisson_Distribution_Stefans_diss_Threshold_4.25.txt", names=["nSpots", "rel. counts"])
N_spots_hist_threshold[4.50] = get_splined("Check_Poisson_Distribution_Stefans_diss_Threshold_4.50.txt", names=["nSpots", "rel. counts"])
N_spots_hist_threshold[4.75] = get_splined("Check_Poisson_Distribution_Stefans_diss_Threshold_4.75.txt", names=["nSpots", "rel. counts"])
N_spots_hist_threshold[5.00] = get_splined("Check_Poisson_Distribution_Stefans_diss_Threshold_5.00.txt", names=["nSpots", "rel. counts"])
N_spots_hist_threshold[5.25] = get_splined("Check_Poisson_Distribution_Stefans_diss_Threshold_5.25.txt", names=["nSpots", "rel. counts"])
N_spots_hist_threshold[5.50] = get_splined("Check_Poisson_Distribution_Stefans_diss_Threshold_5.50.txt", names=["nSpots", "rel. counts"])
N_spots_hist_threshold[5.75] = get_splined("Check_Poisson_Distribution_Stefans_diss_Threshold_5.75.txt", names=["nSpots", "rel. counts"])
N_spots_hist_threshold[6.00] = get_splined("Check_Poisson_Distribution_Stefans_diss_Threshold_6.00.txt", names=["nSpots", "rel. counts"])
N_spots_hist_threshold[6.25] = get_splined("Check_Poisson_Distribution_Stefans_diss_Threshold_6.25.txt", names=["nSpots", "rel. counts"])
N_spots_hist_threshold[6.50] = get_splined("Check_Poisson_Distribution_Stefans_diss_Threshold_6.50.txt", names=["nSpots", "rel. counts"])
N_spots_hist_threshold[6.75] = get_splined("Check_Poisson_Distribution_Stefans_diss_Threshold_6.75.txt", names=["nSpots", "rel. counts"])

bgd_extrapolation = get_splined("HPA_background_extrapolation_Stefan_by_slack.txt", names=["HPA_TS", "rel. counts"])

expectation_vs_threshold = get_splined("Expectation_Stefans_diss_Fig_7_17.txt", names=["-log10(pVal)", "N_spots"])

sensitivity = get_splined("Sensitivity_PS_paper_Fig_10_arxiv_1609_04981.txt", names=["nSources", "flux_per_source"])
discovery_potential = get_splined("DiscPot_PS_paper_Fig_10_arxiv_1609_04981.txt", names=["nSources", "flux_per_source"])
upper_limit_90_per_cent = get_splined("UL_PS_paper_Fig_10_arxiv_1609_04981.txt", names=["nSources", "flux_per_source"]) 
diffuse_flux_example = get_splined("Diffuse_Flux_Example_PS_paper_Fig_10_arxiv_1609_04981.txt", names=["nSources", "flux_per_source"])

TS_parametrization_eta = get_splined("TS_parametrization_NDoF_Stefan.txt", names=["sinDec", "eta"])
TS_parametrization_NDoF = get_splined("TS_parametrization_NDoF_Stefan.txt", names=["sinDec", "NDoF"])
 
limit_E2 = get_splined("Sensitivity_E_2_7yr_PS_paper.txt", names=["sinDec", "flux"])
sensitivity_E3 = get_splined("Sensitivity_E_3_7yr_PS_paper.txt", names=["sinDec", "flux"])
discovery_potential_E3 = get_splined("DiscPot_E_3_7yr_PS_paper.txt", names=["sinDec", "flux"])
