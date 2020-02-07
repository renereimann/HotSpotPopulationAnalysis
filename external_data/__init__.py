import numpy as np
from scipy.interpolate import UnivariateSpline
from ps_analysis.util.digitizer import digitizer
# 


from .Check_Poisson_Distribution_Stefans_diss import N_spots_hist_threshold as _N_spots_hist_threshold
N_spots_hist_threshold = {}
for k in _N_spots_hist_threshold.keys():
    N_spots_hist_threshold[k] = digitizer(_N_spots_hist_threshold[k], clip=True)

_expectation_vs_threshold_digitized = np.genfromtxt("Expectation_Stefans_diss_Fig_7_17.txt", delimiter=", ", comments="#", names=["-log10(pVal)", "N_spots"])
expectation_vs_threshold = UnivariateSpline(_expectation_vs_threshold_digitized["-log10(pVal)"], _expectation_vs_threshold_digitized["N_spots"], s=0, k=1)

from .Sensitivity_Disc_pot_UL_PS_paper_Fig_10_arxiv_1609_04981 import sensitivity_digitized as _sensitivity_digitized
from .Sensitivity_Disc_pot_UL_PS_paper_Fig_10_arxiv_1609_04981 import discovery_potential_digitized as _discovery_potential_digitized 
from .Sensitivity_Disc_pot_UL_PS_paper_Fig_10_arxiv_1609_04981 import upper_limit_90_per_cent_digitized as _upper_limit_90_per_cent_digitized
from .Sensitivity_Disc_pot_UL_PS_paper_Fig_10_arxiv_1609_04981 import diffuse_flux_example_digitized as _diffuse_flux_example_digitized
sensitivity = digitizer(_sensitivity_digitized, clip=True)
discovery_potential = digitizer(_discovery_potential_digitized, clip=True)
upper_limit_90_per_cent = digitizer(_upper_limit_90_per_cent_digitized, clip=True)
diffuse_flux_example = digitizer(_diffuse_flux_example_digitized, clip=True)


from .HPA_intermediat_plot_from_stefan_per_slack import bgd_extrapolation as _bgd_extrapolation
bgd_extrapolation = digitizer(_bgd_extrapolation, clip=True)


_TS_parametrization_digitized_eta = np.genfromtxt("TS_parametrization_NDoF_Stefan.txt", delimiter=", ", comments="#", names=["sinDec", "eta"])
TS_parametrization_eta = UnivariateSpline(_TS_parametrization_digitized_eta["sinDec"], _TS_parametrization_digitized_eta["eta"], s=0, k=1)

_TS_parametrization_digitized_NDoF = np.genfromtxt("TS_parametrization_NDoF_Stefan.txt", delimiter=", ", comments="#", names=["sinDec", "NDoF"])
TS_parametrization_NDoF = UnivariateSpline(_TS_parametrization_digitized_NDoF["sinDec"], _TS_parametrization_digitized_NDoF["NDoF"], s=0, k=1)

_limit_E_2_digitized = np.genfromtxt("Sensitivity_E_2_7yr_PS_paper.txt", delimiter=", ", comments="#", names=["sinDec", "flux"])
limit_E2 = UnivariateSpline(_limit_E_2_digitized["sinDec"], _limit_E_2_digitized["flux"], s=0, k=1 )

_sensitivity_E_3_digitized = np.genfromtxt("Sensitivity_E_3_7yr_PS_paper.txt", delimiter=", ", comments="#", names=["sinDec", "flux"])
sensitivity_E3 = UnivariateSpline(_sensitivity_E_3_digitized["sinDec"], _sensitivity_E_3_digitized["flux"], s=0, k=1)

_discovery_potential_E_3_digitized = np.genfromtxt("DiscPot_E_3_7yr_PS_paper.txt", delimiter=", ", comments="#", names=["sinDec", "flux"])
discovery_potential_E3 = UnivariateSpline(_discovery_potential_E_3_digitized["sinDec"], _discovery_potential_E_3_digitized["flux"], s=0, k=1)
