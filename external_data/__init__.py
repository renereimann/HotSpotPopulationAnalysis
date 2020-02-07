import numpy as np

from ps_analysis.util.digitizer import digitizer

from .Check_Poisson_Distribution_Stefans_diss import N_spots_hist_threshold as _N_spots_hist_threshold
from .Expectation_Stefans_diss_Fig_7_17 import expectation_vs_threshold_digitized as _expectation_vs_threshold_digitized
from .Sensitivity_Disc_pot_UL_PS_paper_Fig_10_arxiv_1609_04981 import sensitivity_digitized as _sensitivity_digitized
from .Sensitivity_Disc_pot_UL_PS_paper_Fig_10_arxiv_1609_04981 import discovery_potential_digitized as _discovery_potential_digitized 
from .Sensitivity_Disc_pot_UL_PS_paper_Fig_10_arxiv_1609_04981 import upper_limit_90_per_cent_digitized as _upper_limit_90_per_cent_digitized
from .Sensitivity_Disc_pot_UL_PS_paper_Fig_10_arxiv_1609_04981 import diffuse_flux_example_digitized as _diffuse_flux_example_digitized
_TS_parametrization_digitized_eta = np.genfromtxt("TS_parametrization_NDoF_Stefan.txt", delimiter=", ", comments="#", names=["sinDec", "eta"])
_TS_parametrization_digitized_NDoF = np.genfromtxt("TS_parametrization_NDoF_Stefan.txt", delimiter=", ", comments="#", names=["sinDec", "NDoF"])
_limit_E_2_digitized = np.genfromtxt("Sensitivity_E_2_7yr_PS_paper.txt", delimiter=", ", comments="#", names=["sinDec", "flux"])
_sensitivity_E_3_digitized = np.genfromtxt("Sensitivity_E_3_7yr_PS_paper.txt", delimiter=", ", comments="#", names=["sinDec", "flux"])
_discovery_potential_E_3_digitized = np.genfromtxt("DiscPot_E_3_7yr_PS_paper.txt", delimiter=", ", comments="#", names=["sinDec", "flux"])

from .HPA_intermediat_plot_from_stefan_per_slack import bgd_extrapolation as _bgd_extrapolation

N_spots_hist_threshold = {}
for k in _N_spots_hist_threshold.keys():
    N_spots_hist_threshold[k] = digitizer(_N_spots_hist_threshold[k], clip=True)
    
expectation_vs_threshold = digitizer(_expectation_vs_threshold_digitized, clip=True)

sensitivity = digitizer(_sensitivity_digitized, clip=True)
discovery_potential = digitizer(_discovery_potential_digitized, clip=True)
upper_limit_90_per_cent = digitizer(_upper_limit_90_per_cent_digitized, clip=True)
diffuse_flux_example = digitizer(_diffuse_flux_example_digitized, clip=True)

TS_parametrization_eta = digitizer(_TS_parametrization_digitized_eta, clip=True)
TS_parametrization_NDoF = digitizer(_TS_parametrization_digitized_NDoF, clip=True)

sensitivity_E3 = digitizer(_sensitivity_E_3_digitized, clip=True)
discovery_potential_E3 = digitizer(_discovery_potential_E_3_digitized, clip=True)
bgd_extrapolation = digitizer(_bgd_extrapolation, clip=True)

limit_E2 = digitizer(_limit_E_2_digitized, clip=True)
