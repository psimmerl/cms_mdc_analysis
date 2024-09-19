import os
import sys
import pathlib
import numpy as np

import pickle

import ROOT as rt
from ROOT import RDataFrame, TFile

from src.histo_utilities import std_color_list as SCL

# **************************** #
STAT = "tight"
# LUMI = 1.1328524540090597e-06 * 27.82 * 1000 # ???
LUMI = 1#1.1328524540090597e-06 * 1000 # ???
RUN2_BR = 2.16e-03  # Adjust weights to Run 2 BR limit
YEAR = "2023"

NODENAME = os.uname().nodename
if "lxplus" in NODENAME:
    LOCAL_DIR = "/eos/user/f/fernanpe/mds_analysis_Christina/"
    FN_MC = f"{LOCAL_DIR}/data/processed/mc_pedro_hlt566_2023.root"
    FN_R3 = f"{LOCAL_DIR}/data/processed/r3_pedro_hlt566_2023.root"
elif "psimmerl-LAU248" in NODENAME:
    LOCAL_DIR = "/home/psimmerl/mds_analysis"
    FN_MC = f"{LOCAL_DIR}/data/processed/mc_pedro_hlt566.root"
    FN_R3 = f"{LOCAL_DIR}/data/processed/r3_pedro_hlt566.root"
else:
    LOCAL_DIR = "/home/psimmerl/mds_analysis"
    FN_MC = f"{LOCAL_DIR}/data/processed/mc_pedro_hlt566.root"
    FN_R3 = f"{LOCAL_DIR}/data/processed/r3_pedro_hlt566.root"

OUT_DIR = f"{LOCAL_DIR}/reports/weekly/2024-04-15"


FN_MC = f"{LOCAL_DIR}/data/raw/mc_2022.root"
FN_R3 = f"{LOCAL_DIR}/data/raw/data_2022.root"


# STAT = "raw"
# LUMI = 23.02 * 1000
# FN_MC = f"{LOCAL_DIR}/data/raw/ggH_HToSSTobbbb_MH-125_MS-15_CTau1000_13p6TeV_1pb_weighted_v6.root"
# FN_R3 = f"{LOCAL_DIR}/data/raw/DisplacedJet-EXOCSCCluster_Run2022EFG-PromptReco-v1_goodLumi_v6.root"

# **** #
LOW_MET_CUTOFF = 150  # 75
HIGH_MET_CUTOFF = 150

# **** #
CUTS_L1 = [
    "acceptance",
    "HLT",
    "L1",
    "MET",
    #! Reset cutflow indices here
    "CSC0 IT",
    "CSC1 IT",
    "DT IT",
    "CSC0 ME1",
    "1 CSC-CSC",
    # "dPhi",
]

CUTS = [
    "MET",
    "acceptance",
    "CSC0 ME1",
    "L1",
    "HLT",
    "0 DT",
    "Exactly 1 CSC-CSC",
    # "1 CSC-CSC",
    #! Reset cutflow indices here
    # "CSC IT", # breaks oot
    "CSC0 IT",
    "CSC1 IT",
    # "ME1",
    "CSC1 ME1",
    # "CSC1 not CSC0",
    #! Reset cutflow indices here
    "dPhi $>$ 1.8",
    # "DT IT",
    # "n leptons",
    # "n jets",
    "CSC muon veto", 
    # "DT muon veto",
    # "CSC jet veto", 
    # "CSC jet veto $<$ 50", 
    # "CSC jet veto $<$ 30", 
    "CSC jet veto $<$ 10", 
    # "DT jet veto",
    # "halo veto",
    # "MB1",
    # "DT stn",
    # "BDT",
    "DNN $>$ 0.96",
    # "dR",
    # "dEta",
    # "0 DT",
    # "1 CSC-CSC",
    # "dPhi $>$ 1.8",
]

# **** #
CUT_VALUES = {
    "l1_lt200" : {
        "MIN_CSC_TIME": -5.0,
        "MAX_CSC_TIME": 12.5,
        "MAX_CSC_TSPREAD": 20.0,
        "MAX_RPC_BX": 0,
        "MIN_RPC_HITS": 1,
        "MAX_CSC_JET": 999,
        "MAX_DT_JET": 999,
        "MAX_CSC_MUON": 999,
        "MAX_DT_MUON": 999,
        "MAX_ME1": 0,
        "MAX_MB1": 0,
        "HALO_CUTOFF": 0,
        "MIN_DPHI": 0,
        "MIN_DETA": 0,
        "MAX_DETA": 4,
        "MIN_CSC_DNN": 0,
        # "MIN_DT_DNN": 0,
    },
    "scs_lt200" : {
        "MIN_CSC_TIME": -5.0,
        "MAX_CSC_TIME": 12.5,
        "MAX_CSC_TSPREAD": 20.0,
        "MAX_RPC_BX": 0,
        "MIN_RPC_HITS": 1,
        "MAX_CSC_JET": 30,
        "MAX_DT_JET": 50,
        "MAX_CSC_MUON": 30,
        "MAX_DT_MUON": 10,
        "MAX_ME1": 0,
        "MAX_MB1": 0,
        "HALO_CUTOFF": 0,
        "MIN_DPHI": 1,
        "MIN_DETA": 0,
        "MAX_DETA": 4,
        "MIN_CSC_DNN": 0,
        # "MIN_DT_DNN": 0,
    },
    "scs_low" : {
        "MIN_CSC_TIME": -5.0,
        "MAX_CSC_TIME": 12.5,
        "MAX_CSC_TSPREAD": 20.0,
        "MAX_RPC_BX": 0,
        "MIN_RPC_HITS": 1,
        "MAX_CSC_JET": 30,
        "MAX_DT_JET": 50,
        "MAX_CSC_MUON": 30,
        "MAX_DT_MUON": 10,
        "MAX_ME1": 0,
        "MAX_MB1": 0,
        "HALO_CUTOFF": 0,
        "MIN_DPHI": 1,
        "MIN_DETA": 0,
        "MAX_DETA": 4,
        "MIN_CSC_DNN": 0,
        # "MIN_DT_DNN": 0,
    },
    "scs_high" : {
        "MIN_CSC_TIME": -5.0,
        "MAX_CSC_TIME": 12.5,
        "MAX_CSC_TSPREAD": 20.0,
        "MAX_RPC_BX": 0,
        "MIN_RPC_HITS": 1,
        "MAX_CSC_JET": 999,
        "MAX_DT_JET": 999,
        "MAX_CSC_MUON": 999,
        "MAX_DT_MUON": 999,
        "MAX_ME1": 0,
        "MAX_MB1": 0,
        "HALO_CUTOFF": 0,
        "MIN_DPHI": 1,
        "MIN_DETA": 0,
        "MAX_DETA": 4,
        "MIN_CSC_DNN": 0,
        # "MIN_DT_DNN": 0,
    },
    "ropt_lt200" : {
        "MIN_CSC_TIME": -5.0,
        "MAX_CSC_TIME": 12.5,
        "MAX_CSC_TSPREAD": 20.0,
        "MAX_RPC_BX": 0,
        "MIN_RPC_HITS": 1,
        "MAX_CSC_JET": 20,
        "MAX_DT_JET": 50,
        "MAX_CSC_MUON": 10,
        "MAX_DT_MUON": 100,
        "MAX_ME1": 0,
        "MAX_MB1": 0,
        "HALO_CUTOFF": 0,
        "MIN_DPHI": 1.8,
        "MIN_DETA": 0,
        "MAX_DETA": 4,
        "MIN_CSC_DNN": 0,
        # "MIN_DT_DNN": 0,
    },
    "ropt_low" : {
        # Optimized with IT S/rt[B]. S=1266, B=267, S/rt[B]=77.5
        "MIN_CSC_TIME": -5.0,
        "MAX_CSC_TIME": 12.5,
        "MAX_CSC_TSPREAD": 20.0,
        "MAX_RPC_BX": 0,
        "MIN_RPC_HITS": 1,
        "MAX_CSC_JET": 10,
        "MAX_DT_JET": 10,
        "MAX_CSC_MUON": 13.6,
        "MAX_DT_MUON": 9.1,
        "MAX_ME1": 0, # HLT requires this to be 0
        "MAX_MB1": 999,
        "HALO_CUTOFF": 0.01, # applied to dt phi
        "MIN_DPHI": 1.8,
        "MIN_DETA": 0,
        "MAX_DETA": 4,
        "MIN_CSC_DNN": 0.96,  # pedro's value, bkgMC_plusBeamHalo: S=, B=, S/rt[B]=
    },
    "ropt_high" : {
        "MIN_CSC_TIME": -5.0,
        "MAX_CSC_TIME": 12.5,
        "MAX_CSC_TSPREAD": 20.0,
        "MAX_RPC_BX": 0,
        "MIN_RPC_HITS": 1,
        "MAX_CSC_JET": 999,
        "MAX_DT_JET": 999,
        "MAX_CSC_MUON": 999,
        "MAX_DT_MUON": 999,
        "MAX_ME1": 0,
        "MAX_MB1": 0,
        "HALO_CUTOFF": 0.4,
        "MIN_DPHI": 1.8,
        "MIN_DETA": 0,
        "MAX_DETA": 4,
        "MIN_CSC_DNN": 0.96,  # pedro's value, bkgMC_plusBeamHalo: S=, B=, S/rt[B]=
    },
    "tight_low" : {
        "MIN_CSC_TIME": -5.0,
        "MAX_CSC_TIME": 12.5,
        "MAX_CSC_TSPREAD": 20.0,
        "MAX_RPC_BX": 0,
        "MIN_RPC_HITS": 1,
        "MAX_CSC_JET": 10,
        "MAX_DT_JET": 10,
        "MAX_CSC_MUON": 10,
        "MAX_DT_MUON": 10,
        "MAX_ME1": 0,
        "MAX_MB1": 0,
        "HALO_CUTOFF": 0.4,
        "MIN_DPHI": 1.8,
        "MIN_DETA": 0,
        "MAX_DETA": 4,
        "MIN_CSC_DNN": 0.96,
    },
    "tight_high" : {
        "MIN_CSC_TIME": -5.0,
        "MAX_CSC_TIME": 12.5,
        "MAX_CSC_TSPREAD": 20.0,
        "MAX_RPC_BX": 0,
        "MIN_RPC_HITS": 1,
        "MAX_CSC_JET": 999,
        "MAX_DT_JET": 999,
        "MAX_CSC_MUON": 999,
        "MAX_DT_MUON": 999,
        "MAX_ME1": 10,
        "MAX_MB1": 0,
        "HALO_CUTOFF": 0.4,
        "MIN_DPHI": 1.8,
        "MIN_DETA": 0,
        "MAX_DETA": 4,
        "MIN_CSC_DNN": 0.96,  # pedro's value, bkgMC_plusBeamHalo: S=, B=, S/rt[B]=
    },
}

OPT_CUTS = [
    # "MB1",
    "CSC jet veto",
    "DT jet veto",
    "CSC muon veto",
    "DT muon veto",
    "halo veto",
    "CSC DNN"
]

# sort the values in order of preference for when score=score0
CUT_OPT_PARS = { 
    "ME1": {
        "col": "cscNHitME1",
        "cut": "MAX_ME1",
        "values": np.arange(0, 11, 1),
        "func": lambda v, vd: vd <= v,
    },
    "MB1": {
        "col": "dtNHitStation1",
        "cut": "MAX_MB1",
        "values": np.arange(0, 11, 1),
        "func": lambda v, vd: vd <= v,
    },
    "CSC jet veto": {
        "col": "cscJetVetoPt",
        "cut": "MAX_CSC_JET",
        "values": np.arange(10, 201, 1),
        # "func" : lambda v, vd: (10 < vd) & (vd < v),
        "func": lambda v, vd: vd < v,
    },
    "DT jet veto": {
        "col": "dtJetVetoPt",
        "cut": "MAX_DT_JET",
        "values": np.arange(10, 201, 1),
        # "func" : lambda v, vd: (10 < vd) & (vd < v),
        "func": lambda v, vd: vd < v,
    },
    "CSC muon veto": {
        "col": "cscMuonVetoPt",
        "cut": "MAX_CSC_MUON",
        "values": np.arange(0, 201, 1),
        # "func" : lambda v, vd: (0 < vd) & (vd < v),
        "func": lambda v, vd: vd < v,
    },
    "DT muon veto": {
        "col": "dtMuonVetoPt",
        "cut": "MAX_DT_MUON",
        "values": np.arange(0, 201, 1),
        # "func" : lambda v, vd: (0 < vd) & (vd < v),
        "func": lambda v, vd: vd < v,
    },
    "halo veto": {
        "col": "dtPhi",
        "cut": "HALO_CUTOFF",
        "values": np.arange(0, 0.52, 0.02)[::-1],
        "func": lambda v, vd: (v < np.abs(vd)) & (np.abs(vd) < PI - v),
    },
    "min dPhi": {
        "col": "tag_dPhi",
        "cut": "MIN_DPHI",
        "values": np.arange(0, 0.62, 0.02)[::-1],
        "func": lambda v, vd: v < vd,
    },
    "min dEta": {
        "col": "tag_dEta",
        "cut": "MIN_DETA",
        "values": np.arange(0, 4.1, 0.1)[::-1],
        "func": lambda v, vd: v < vd,
    },
    "max dEta": {
        "col": "tag_dEta",
        "cut": "MAX_DETA",
        "values": np.arange(0, 4.1, 0.1),
        "func": lambda v, vd: vd < v,
    },
    "CSC DNN": {
        "col": "cscDNN",
        "cut": "MIN_CSC_DNN",
        "values": np.arange(0, 1.01, 0.01)[::-1],
        "func": lambda v, vd: v < vd,
    },
    # "DT DNN": {
    #     "col": "dtDNN",
    #     "cut": "MIN_DT_DNN",
    #     "values": np.arange(0, 1.01, 0.01)[::-1],
    #     "func": lambda v, vd: v < vd,
    # },
    "ABCD phi": {
        "col": "tag_dPhi",
        "cut": "ABCD_PHI",
        "values": np.arange(0, 3.5, 0.01)[::-1],
        "func": lambda v, vd: v <= vd,
    },
    "ABCD size": {
        "col": "dtSize",
        "cut": "ABCD_SIZE",
        "values": np.arange(50, 150, 1),
        "func": lambda v, vd: v <= vd,
    },
    "DT size": {
        "col": "dtSize",
        "cut": "MIN_DT_SIZE",
        "values": np.arange(50, 150, 1),
        "func": lambda v, vd: v <= vd,
    },
}

# **************************** #
C, D = "cscRechitCluster", "dtRechitCluster"

COLUMNS_OUT = [
    # "Flag2_BadChargedCandidateFilter",
    # "Flag2_BadPFMuonFilter",
    # "Flag2_EcalDeadCellTriggerPrimitiveFilter",
    # "Flag2_HBHEIsoNoiseFilter",
    # "Flag2_HBHENoiseFilter",
    # "Flag2_all",
    # "Flag2_ecalBadCalibFilter",
    # "Flag2_eeBadScFilter",
    # "Flag2_globalSuperTightHalo2016Filter",
    # "Flag2_globalTightHalo2016Filter",
    # "Flag_BadChargedCandidateFilter",
    # "Flag_BadPFMuonFilter",
    # "Flag_CSCTightHaloFilter",
    # "Flag_HBHEIsoNoiseFilter",
    # "Flag_HBHENoiseFilter",
    # "Flag_all",
    # "Flag_ecalBadCalibFilter",
    # "Flag_eeBadScFilter",
    # "Flag_globalSuperTightHalo2016Filter",
    # "Flag_goodVertices",
    # "HLTDecision",
    # "MC_condition",
    #
    "cscRechitClusterAvgStation10",
    "cscRechitClusterEta",
    "cscRechitClusterJetVetoE",
    "cscRechitClusterJetVetoLooseId",
    "cscRechitClusterJetVetoPt",
    "cscRechitClusterJetVetoTightId",
    "cscRechitClusterMaxChamber",
    "cscRechitClusterMaxChamberRatio",
    "cscRechitClusterMaxStation",
    "cscRechitClusterMaxStationRatio",
    "cscRechitClusterMet_dPhi",
    "cscRechitClusterMuonVetoE",
    "cscRechitClusterMuonVetoGlobal",
    "cscRechitClusterMuonVetoLooseId",
    "cscRechitClusterMuonVetoPt",
    "cscRechitClusterNChamber",
    "cscRechitClusterNRechitChamberMinus11",
    "cscRechitClusterNRechitChamberMinus12",
    "cscRechitClusterNRechitChamberMinus13",
    "cscRechitClusterNRechitChamberMinus21",
    "cscRechitClusterNRechitChamberMinus22",
    "cscRechitClusterNRechitChamberMinus31",
    "cscRechitClusterNRechitChamberMinus32",
    "cscRechitClusterNRechitChamberMinus41",
    "cscRechitClusterNRechitChamberMinus42",
    "cscRechitClusterNRechitChamberPlus11",
    "cscRechitClusterNRechitChamberPlus12",
    "cscRechitClusterNRechitChamberPlus13",
    "cscRechitClusterNRechitChamberPlus21",
    "cscRechitClusterNRechitChamberPlus22",
    "cscRechitClusterNRechitChamberPlus31",
    "cscRechitClusterNRechitChamberPlus32",
    "cscRechitClusterNRechitChamberPlus41",
    "cscRechitClusterNRechitChamberPlus42",
    "cscRechitClusterNStation10",
    "cscRechitClusterPhi",
    "cscRechitClusterSize",
    "cscRechitClusterTime",
    "cscRechitClusterTimeSpread",
    "cscRechitClusterTimeSpreadWeightedAll",
    "cscRechitClusterTimeWeighted",
    "cscRechitClusterX",
    "cscRechitClusterY",
    "cscRechitClusterZ",
    "cscRechitCluster_match_MB1Seg_0p4",
    "cscRechitCluster_match_RB1_0p4",
    "cscRechitCluster_match_RE12_0p4",
    "cscRechitCluster_match_dtSeg_0p4",
    #"cscRechitClusterDNN_bkgMC",
    #"cscRechitClusterDNN_bkgMC_plusBeamHalo",
    #"cscRechitClusterDNN_bkgOOTData",
    # "cscRechitCluster_match_gLLP",
    # "cscRechitCluster_match_gLLP_csc",
    # "cscRechitCluster_match_gLLP_decay_r",
    # "cscRechitCluster_match_gLLP_decay_z",
    # "cscRechitCluster_match_gLLP_dt",
    # "cscRechitCluster_match_gLLP_e",
    # "cscRechitCluster_match_gLLP_eta",
    # "cscRechitCluster_match_gLLP_index",
    # "cscRechitCluster_match_gLLP_minDeltaR",
    # "cscRechitCluster_match_gLLP_phi",
    #
    # "ctau",
    #
    # "dtRechitClusterAvgStation10",
    # "dtRechitClusterEta",
    # "dtRechitClusterJetVetoE",
    # "dtRechitClusterJetVetoLooseId",
    # "dtRechitClusterJetVetoPt",
    # "dtRechitClusterJetVetoTightId",
    # "dtRechitClusterMaxChamber",
    # "dtRechitClusterMaxChamberRatio",
    # "dtRechitClusterMaxStation",
    # "dtRechitClusterMaxStationRatio",
    # "dtRechitClusterMet_dPhi",
    # "dtRechitClusterMuonVetoE",
    # "dtRechitClusterMuonVetoGlobal",
    # "dtRechitClusterMuonVetoLooseId",
    # "dtRechitClusterMuonVetoPt",
    # "dtRechitClusterMuonVetoTightId",
    # "dtRechitClusterNChamber",
    # "dtRechitClusterNHitStation1",
    # "dtRechitClusterNHitStation2",
    # "dtRechitClusterNHitStation3",
    # "dtRechitClusterNHitStation4",
    # "dtRechitClusterNOppositeSegStation1",
    # "dtRechitClusterNOppositeSegStation2",
    # "dtRechitClusterNOppositeSegStation3",
    # "dtRechitClusterNOppositeSegStation4",
    # "dtRechitClusterNSegStation1",
    # "dtRechitClusterNSegStation2",
    # "dtRechitClusterNSegStation3",
    # "dtRechitClusterNSegStation4",
    # "dtRechitClusterNStation10",
    # "dtRechitClusterNoiseHit",
    # "dtRechitClusterNoiseHitStation1",
    # "dtRechitClusterNoiseHitStation2",
    # "dtRechitClusterNoiseHitStation3",
    # "dtRechitClusterNoiseHitStation4",
    # "dtRechitClusterOverlap",
    # "dtRechitClusterPhi",
    # "dtRechitClusterSize",
    # "dtRechitClusterWheel",
    # "dtRechitClusterX",
    # "dtRechitClusterY",
    # "dtRechitClusterZ",
    # "dtRechitCluster_match_MB1Seg_0p4",
    # "dtRechitCluster_match_MB1Seg_0p5",
    # "dtRechitCluster_match_MB1hits_0p4",
    # "dtRechitCluster_match_MB1hits_0p5",
    # "dtRechitCluster_match_MB1hits_cosmics_minus",
    # "dtRechitCluster_match_MB1hits_cosmics_plus",
    # "dtRechitCluster_match_RB1_0p4",
    # "dtRechitCluster_match_RB1_dPhi0p5",
    # "dtRechitCluster_match_RPCBx_dPhi0p5",
    # "dtRechitCluster_match_RPChits_dPhi0p5",
    # # "dtRechitCluster_match_gLLP",
    # # "dtRechitCluster_match_gLLP_csc",
    # # "dtRechitCluster_match_gLLP_decay_r",
    # # "dtRechitCluster_match_gLLP_decay_z",
    # # "dtRechitCluster_match_gLLP_dt",
    # # "dtRechitCluster_match_gLLP_e",
    # # "dtRechitCluster_match_gLLP_eta",
    # # "dtRechitCluster_match_gLLP_index",
    # # "dtRechitCluster_match_gLLP_minDeltaR",
    # # "dtRechitCluster_match_gLLP_phi",
    #
    "evtNum",
    # "gHiggsE",
    # "gHiggsEta",
    # "gHiggsPhi",
    # "gHiggsPt",
    # "gLLP_beta",
    # "gLLP_csc",
    # "gLLP_ctau",
    # "gLLP_decay_vertex_r",
    # "gLLP_decay_vertex_x",
    # "gLLP_decay_vertex_y",
    # "gLLP_decay_vertex_z",
    # "gLLP_dt",
    # "gLLP_e",
    # "gLLP_eta",
    # "gLLP_phi",
    # "gLLP_pt",
    "jetE",
    "jetEta",
    "jetPhi",
    "jetPt",
    "jetTightPassId",
    "lepDZ",
    "lepE",
    "lepEta",
    "lepPassLooseIso",
    "lepPassTightIso",
    "lepPassVTightIso",
    "lepPassVVTightIso",
    "lepPdgId",
    "lepPhi",
    "lepPt",
    "lepTightId",
    # "lumiSec",
    # "mH",
    # "mX",
    "met",
    "metPhi",
    "nCscRechitClusters",
    # "nCscRings",
    "nDtRechitClusters",
    # "nDtRings",
    # "nGLLP",
    "nJets",
    "nLeptons",
    # "npu",
    # "npv",
    # "pileupWeight",
    # "rho",
    # "runNum",
    "weight",
]

# **** #
pathlib.Path(OUT_DIR).mkdir(parents=True, exist_ok=True)

rng = np.random.default_rng()

rt.gErrorIgnoreLevel = 1001  # rt.kInfo + 1
rt.gROOT.SetBatch(True)
PI = rt.TMath.Pi()

gc = []  # ROOT garbage collector

# **************************** #
if __name__ == "__main__":
    print("+-------------------------+")
    print("| Starting skim_csccsc.py |")
    print("+-------------------------+")

    # rt.EnableImplicitMT(16)
    # print("    Enabled ROOT's implicit multithreading (sometimes causes a crash)")

    # **************************** #
    N_ITERATIONS = 1
    OPT_SCORES = []
    OPT_FRAC = 1.0  # fraction with replacement, if 1 just bootstrap
    LOO = False
    RAND = False
    accepted_cut = False

    PRINT_CUTFLOW = False
    MET_CATEGORY = ("lt200", "low", "high")[0]
    CUTSET = "scs"
    OOT = False
    ITVal1 = False
    ITVal2 = False

    DNN_VERSION = (None, "bkgMC", "bkgMC_plusBeamHalo", "bkgOOTData")[2]

    args = " " + (" ".join(sys.argv[1:]) if len(sys.argv) > 1 else "") + " "

    # surround with spaces so you get who word matching (eg 'cutflow' contains 'low' )
    if " cutflow " in args.lower():
        print("    Printing cutflow tables")
        PRINT_CUTFLOW = True

    if " l1 " in args:
        print("    Using the reduced cut set (l1)")
        CUTS = CUTS_L1
        CUTSET = "l1"

    if " low " in args:  # otherwise cutflow breaks it
        print("    Low met category")
        CUTS = [c.replace("MET", "low MET") if "MET" == c else c for c in CUTS]
        MET_CATEGORY = "low"
    elif " high " in args:
        print("    High met category")
        print("        REMOVING HALO, JET, MUON VETOS")
        CUTS = [c.replace("MET", "high MET") if "MET" == c else c for c in CUTS]
        CUTS = [c for c in CUTS if "halo" not in c]
        CUTS = [c for c in CUTS if "jet" not in c]
        CUTS = [c for c in CUTS if "muon" not in c]
        OPT_CUTS = [c for c in OPT_CUTS if "halo" not in c]
        OPT_CUTS = [c for c in OPT_CUTS if "jet" not in c]
        OPT_CUTS = [c for c in OPT_CUTS if "muon" not in c]
        print("        REMOVING CSC1 ME1")
        CUTS = [c for c in CUTS if "CSC1 ME1" not in c]
        print("        REMOVING DNN")
        CUTS = [c for c in CUTS if "DNN" not in c]
        OPT_CUTS = [c for c in OPT_CUTS if "DNN" not in c]
        MET_CATEGORY = "high"
    else:
        print("    No met categorization (only met<200)")
        MET_CATEGORY = "lt200"

    if " roptDNN " in args: # I removed the separate cutset since we don"t optimize both
        print("    Using the randomly optimized cut set with DNN (roptDNN)")
        CUTSET = "ropt"
    elif " ropt " in args:
        print("    Using the randomly optimized cut set (ropt)")
        CUTSET = "ropt"
    elif " tightDNN " in args: # I removed the separate cutset since we don't optimize both
        print("    Using the tight cut set with DNN (tightDNN)")
        CUTSET = "tight"
    elif " tight " in args:
        print("    Using the tight cut set (tight)")
        CUTSET = "tight"
    elif " loptDNN " in args:
        print("    Using the leave-one-out optimized cut set with DNN (loptDNN)")
        CUTSET = "loptDNN"
        raise DeprecationWarning("lopt/LOO was removed, use random opt (ropt/RAND)")
    elif " lopt " in args:
        print("    Using the leave-one-out optimized cut set (lopt)")
        CUTSET = "lopt"
        raise DeprecationWarning("lopt/LOO was removed, use random opt (ropt/RAND)")
    else:
        print("    Using the standard cut set (scs)")
        CUTSET = "scs"

    if " oot " in args:
        print("    Using out-of-time 2nd cluster")
        CUTS = [c.replace("CSC1 IT", "CSC1 OOT") if "CSC1 IT" == c else c for c in CUTS]
        OOT = True
    else:
        print("    Using in-time 2nd cluster")

    if " ITVal1 " in args:
        print("    Reverting the DNN selection")
        CUTS = [c.replace("DNN $>$ 0.96", "DNN $<$ 0.96") if "DNN $>$ 0.96" == c else c for c in CUTS]
        ITVal1 = True

    if " ITVal2 " in args:
        print("    Reverting the jet veto")
        CUTS = [c.replace("CSC jet veto", "CSC jet req") if "CSC jet veto" == c else c for c in CUTS]
        ITVal2 = True

    if " loo " in args:
        print("    PERFORMING LOO OPTIMIZATION")
        if OOT:
            print("        FORCING MC TO IN-TIME")
        LOO = True
        N_ITERATIONS = 501
        raise DeprecationWarning("lopt/LOO was removed, use random opt (ropt/RAND)")
    if " rand " in args:
        print("    PERFORMING RAND OPTIMIZATION")
        if OOT:
            print("        FORCING MC TO IN-TIME")
        RAND = True
        N_ITERATIONS = 501#1001
    # else:
    #     print("    REMOVING DT SIZE CUT")
    #     CUTS = [c for c in CUTS if "DT size" not in c]

    #if "DNN" not in CUTSET:
    # if "dnn" not in args.lower(): # note this isnt a whole word match
    #     print("    Removing DNN from CUTS")
    #     CUTS = [c for c in CUTS if "DNN" not in c]
    #     OPT_CUTS = [c for c in OPT_CUTS if "DNN" not in c]
    #     DNN_VERSION = None
    # else:
    #     if RAND or LOO:
    #         print("        OPTIMIZING JUST DNN")
    #         OPT_CUTS = [c for c in OPT_CUTS if "DNN" in c]

    if " bkgMC_plusBeamHalo " in args:
        print("    Using DNN version bkgMC_plusBeamHalo")
        DNN_VERSION = "bkgMC_plusBeamHalo"
    elif " bkgMC " in args:
        print("    Using DNN version bkgMC")
        DNN_VERSION = "bkgMC"
        CUT_OPT_PARS["CSC DNN"]["values"] = np.arange(0.830, 1.001, 0.001)[::-1]
    elif " bkgOOTData " in args:
        print("    Using DNN version bkgOOTData")
        DNN_VERSION = "bkgOOTData"
    if DNN_VERSION is not None:
        CUT_OPT_PARS["CSC DNN"]["col"] = f"cscDNN_{DNN_VERSION}"

    if f"{CUTSET}_{MET_CATEGORY}" in CUT_VALUES:
        CUT_VALUES = CUT_VALUES[f"{CUTSET}_{MET_CATEGORY}"]
    elif f"{CUTSET}_lt200" in CUT_VALUES:
        CUT_VALUES = CUT_VALUES[f"{CUTSET}_lt200"]
        print(f"    No cutset for {MET_CATEGORY=} found, defaulting to 'lt200'")
    else:
        raise ValueError(f"no cutset associated with {CUTSET=} and {MET_CATEGORY=} found")

    # FN_MC = f"{LOCAL_DIR}/data/processed/mc_hlt566_{YEAR}.root"
    # FN_R3 = f"{LOCAL_DIR}/data/processed/r3_hlt566_{YEAR}.root"
    if " 2022 " in args:
        YEAR, LUMI = "2022", 48.58 * 23.02 * 1000
        FN_MC = f"{LOCAL_DIR}/data/raw/mc_{YEAR}.root"
        FN_R3 = f"{LOCAL_DIR}/data/raw/data_{YEAR}.root"
        STAT += f"_{YEAR}"
        print(f"    Setting {YEAR=}")
        with TFile(FN_MC, "read") as f:
            LUMI /= f.Get("NEvents").GetBinContent(1)
    elif " 2023 " in args:
        YEAR, LUMI = "2023", 48.58 * 27.82 * 1000
        FN_MC = f"{LOCAL_DIR}/data/raw/mc_{YEAR}.root"
        FN_R3 = f"{LOCAL_DIR}/data/raw/data_{YEAR}.root"
        STAT += f"_{YEAR}"
        print(f"    Setting {YEAR=}")
        with TFile(FN_MC, "read") as f:
            LUMI /= f.Get("NEvents").GetBinContent(1)
    print(f"    Luminosity = {f'{LUMI=:.3e}' if LUMI < 1 else f'{LUMI:.2f}'}")

    print("", flush=True)

    # **** #
    # if "1 CSC-CSC" not in CUTS:
    #     raise NotImplementedError("cant handle multiple pairs yet")

    # **************************** #
    for iopt in range(N_ITERATIONS):
        # if (iopt%2) != 0 and iopt != N_ITERATIONS - 1 and (LOO or RAND):
        # if iopt != 0 and iopt != N_ITERATIONS - 1 and (LOO or RAND):
        if (iopt%100 != 0) and iopt != N_ITERATIONS - 1 and (LOO or RAND) and not accepted_cut:
            OPT_CUT = rng.choice([c for c in OPT_CUTS if c != OPT_CUT or len(OPT_CUTS) == 1])
        else:
            OPT_CUT = ""

        if RAND and OPT_CUT != "":
            cn = CUT_OPT_PARS[OPT_CUT]["cut"]
            cvals = CUT_OPT_PARS[OPT_CUT]["values"]
            val0 = CUT_VALUES[cn]
            nc, pvals = len(cvals), np.ones_like(cvals, float) # initially no prior
            ic, ic0 = np.arange(nc), np.searchsorted(cvals, val0)
            if (ic<ic0).sum() and ic0<nc:
                pvals = (ic<ic0)*(nc-(ic<ic0).sum())/(ic<ic0).sum() + ~(ic<ic0) # 50% chance tighter/looser
            pvals /= pvals.sum()
            val = rng.choice(cvals, p=pvals)
            CUT_VALUES[cn] = val
            # for i in range(len(CUT_OPT_PARS[OPT_CUT]["values"]) - 1): # doesn't really help
            # val = rng.choice([v for v in CUT_OPT_PARS[OPT_CUT]["values"] if v not in skip_vals])
            # cv_history = [x[-1] for x in OPT_SCORES]
            # skip_vals = [val0]
            #     if np.prod([[CUT_VALUES[cv_k] == cv_v for cv_k, cv_v in cv.items()] for cv in cv_history], 1).any():
            #         print("skipping already tested!", cn, val, flush=True)
            #         skip_vals.append(val)
            #     else:
            #         break
            # if len(skip_vals) == len(CUT_OPT_PARS[OPT_CUT]["values"]):
            #     print("all test exhausted", cn, flush=True)
            #     continue

        # **************************** #
        MIN_CSC_TIME = CUT_VALUES["MIN_CSC_TIME"]
        MAX_CSC_TIME = CUT_VALUES["MAX_CSC_TIME"]
        MAX_CSC_TSPREAD = CUT_VALUES["MAX_CSC_TSPREAD"]
        MAX_RPC_BX = CUT_VALUES["MAX_RPC_BX"]
        MIN_RPC_HITS = CUT_VALUES["MIN_RPC_HITS"]
        MAX_CSC_JET = CUT_VALUES["MAX_CSC_JET"]
        MAX_DT_JET = CUT_VALUES["MAX_DT_JET"]
        MAX_CSC_MUON = CUT_VALUES["MAX_CSC_MUON"]
        MAX_DT_MUON = CUT_VALUES["MAX_DT_MUON"]
        MAX_ME1 = CUT_VALUES["MAX_ME1"]
        MAX_MB1 = CUT_VALUES["MAX_MB1"]
        HALO_CUTOFF = CUT_VALUES["HALO_CUTOFF"]
        MIN_DPHI = CUT_VALUES["MIN_DPHI"]
        MIN_DETA = CUT_VALUES["MIN_DETA"]
        MAX_DETA = CUT_VALUES["MAX_DETA"]
        MIN_CSC_DNN = CUT_VALUES["MIN_CSC_DNN"]
        # MIN_DT_DNN = CUT_VALUES["MIN_DT_DNN"]
        # ABCD_DPHI = CUT_VALUES["ABCD_DPHI"]
        # ABCD_SIZE = CUT_VALUES["ABCD_SIZE"]

        # **************************** #
        rdfs = {
            "mc" : RDataFrame("MuonSystem", FN_MC),
            "r3" : RDataFrame("MuonSystem", FN_R3),
        }

        if iopt == 0:
            print("Events in:", flush=True)
        for key, rdf in rdfs.items():
            # rdf = rdf.Range(0,100_000) # Skim a subset of events for debugging

            if key == "mc":# and LUMI < 1: # fix weights
                # print(key, LUMI, rdf.Sum("weight").GetValue(), flush=True)
                rdf = rdf.Redefine("weight", f"weight * {LUMI}")
                print("\t", rdf.Sum("weight").GetValue())

            count, weight = rdf.Count().GetValue(), rdf.Sum("weight").GetValue()
            if iopt == 0:
                print(f"  {key} = {count:,} ({weight:,.2f}) -- read", flush=True)

            # **** #
            # Create dummy columns to store what cluster indices pass our selections
            rdf = rdf.Define(f"{C}0Size", f"{C}Size")
            rdf = rdf.Define(f"{C}1Size", f"{C}Size")

            rdf = rdf.Define("evtFlag", "weight > 0")
            rdf = rdf.Define(f"{C}0Flag", f"{C}0Size > 0")
            rdf = rdf.Define(f"{C}1Flag", f"{C}1Size > 0")
            rdf = rdf.Define(f"{D}Flag", f"{D}Size > 0")

            rdf = rdf.Define("evtCutFlag", "weight > 0")
            rdf = rdf.Define(f"{C}0CutFlag", f"{C}0Size > 0")
            rdf = rdf.Define(f"{C}1CutFlag", f"{C}1Size > 0")
            rdf = rdf.Define(f"{D}CutFlag", f"{D}Size > 0")

            # **** #
            # Add cluster radius columns
            for p in (C, D):
                rdf = rdf.Define(f"{p}R", f"ROOT::VecOps::sqrt( {p}X*{p}X + {p}Y*{p}Y )")
                if f"{p}R" not in COLUMNS_OUT:
                    COLUMNS_OUT.append(f"{p}R")

            rdf = rdf.Define(
                f"{C}NHitME1",
                f"{C}NRechitChamberPlus11 + {C}NRechitChamberPlus12 + {C}NRechitChamberMinus11 + {C}NRechitChamberMinus12",
            )
            if f"{C}NHitME1" not in COLUMNS_OUT:
                COLUMNS_OUT.append(f"{C}NHitME1")

            # **** #
            rdfs[key] = rdf

        # #! Find original weights for optimization (will be overwritten when/if I reset indices)
        # wmc_nocuts = rdfs["mc"].Sum("weight").GetValue()
        # wr3_nocuts = rdfs["r3"].Sum("weight").GetValue()

        if iopt == 0:
            print("", flush=True)


        if " n-1 " in args.lower():
            h_list = []
            
            # Other variables apart from the ones built in rdf (listed here) can be filled.
            # use print(rdf.GetColumnNames()) to get the list of variables that can be plotted. Input variables like cscRechitCluster_X are accesible.
            variables = {
                "met": [20, 0, 200],
                "tag_dPhi": [20, 0, 3.5],
                "tag_dR": [20, 0, 5],
                "csc0Met_dPhi": [20, 0, 3.5],
                "csc0Phi": [20, -3.2, 3.2],
                "csc1Phi": [20, -3.2, 3.2],
                "csc0Eta": [20, -2.4, 2.4],
                "csc1Eta": [20, -2.4, 2.4],
                "csc0Size": [20, 50, 800],
                "csc1Size": [20, 50, 800],
                "csc0JetVetoPt": [20, 10, 200],
                "csc1JetVetoPt": [20, 10, 200],
                "csc0MuonVetoPt": [20, 10, 200],
                "csc1MuonVetoPt": [20, 10, 200],
                "csc0TimeSpread": [20, 0, 80],
                "csc1TimeSpread": [20, 0, 80],
                "csc0NStation10": [5, 0, 5],
                "csc1NStation10": [5, 0, 5],
                "csc0DNN_bkgMC_plusBeamHalo": [20, 0, 1],
                "csc1DNN_bkgMC_plusBeamHalo": [20, 0, 1],
                }
            
            
        # **************** #
        for key, rdf in rdfs.items():
            if PRINT_CUTFLOW:
                print(f"{key} cutflow:", flush=True)
                print(r"\begin{center}", flush=True)
                # print(r"\begin{tabular}{c|ccc|ccc|ccc|ccc}", flush=True)
                print(r"\begin{tabular}{c|cc|cc|cc|cc}", flush=True)
                ec0,cc00,cc10,dc0 = rdf.Filter("evtFlag").Sum("weight"), rdf.Filter("evtFlag").Sum(f"{C}0Flag"), rdf.Filter("evtFlag").Sum(f"{C}1Flag"), rdf.Filter("evtFlag").Sum(f"{D}Flag")
                ec0,cc00,cc10,dc0 = ec0.GetValue(),cc00.GetValue(),cc10.GetValue(),dc0.GetValue()
                #fmt: off
                print(r"    \hline", flush=True)
                print(f"    \\textbf{{{'Signal' if 'mc' in key else 'Data'}}}"
                    #   + r" & \multicolumn{3}{c}{CSC-CSC} & \multicolumn{3}{c}{First CSC} & \multicolumn{3}{c}{Second CSC} & \multicolumn{3}{c}{DT} \\", flush=True)
                      + r" & \multicolumn{2}{c}{CSC-CSC} & \multicolumn{2}{c}{First CSC} & \multicolumn{2}{c}{Second CSC} & \multicolumn{2}{c}{DT} \\", flush=True)
                print(r"    \hline", flush=True)
                # print(r"    Selection & N Events & Cut Eff & Cum Eff & N Clusters & Cut Eff & Cum Eff & N Clusters & Cut Eff & Cum Eff & N Clusters & Cut Eff & Cum Eff \\", flush=True)
                print(r"    Selection & Weight & Eff. (\%) & Count & Eff. (\%) & Count & Eff. (\%) & Count & Eff. (\%) \\", flush=True)
                print(r"    \hline", flush=True)
                # print(f"    sample & {ec0:,.0f} & -- & -- & {cc00:,.0f} & -- & -- & {cc10:,.0f} & -- & -- & {dc0:,.0f} & -- & -- \\\\", flush=True)
                print(f"    sample & {ec0:,.0f} & -- & {cc00:,.0f} & -- & {cc10:,.0f} & -- & {dc0:,.0f} & -- \\\\", flush=True)
                #fmt: on

            columns_out = []
            for cut in CUTS:  #! Can I run this in a subprocess?
                # **** #
                # Reset cut flags
                rdf = rdf.Redefine("evtCutFlag", "weight > 0")
                rdf = rdf.Redefine(f"{C}0CutFlag", f"{C}0Size > 0")
                rdf = rdf.Redefine(f"{C}1CutFlag", f"{C}1Size > 0")
                rdf = rdf.Redefine(f"{D}CutFlag", f"{D}Size > 0")

                # **** #
                # Signal matching for MC
                if "acceptance" in cut: 
                    if "mc" in key:
                        rdf = rdf.Redefine(
                            f"{C}0CutFlag", f"{C}0CutFlag && {C}_match_gLLP && "
                            + f"(abs({C}_match_gLLP_eta) < 3) && "
                            + f"({C}_match_gLLP_decay_r < 800) && "
                            + f"(400 < abs({C}_match_gLLP_decay_z)) &&"
                            + f" (abs({C}_match_gLLP_decay_z) < 1200)"
                        )
                        rdf = rdf.Redefine(
                            f"{C}1CutFlag", f"{C}1CutFlag && {C}_match_gLLP && "
                            + f"(abs({C}_match_gLLP_eta) < 3) && "
                            + f"({C}_match_gLLP_decay_r < 800) && "
                            + f"(400 < abs({C}_match_gLLP_decay_z)) &&"
                            + f" (abs({C}_match_gLLP_decay_z) < 1200)"
                        )

                        rdf = rdf.Redefine(
                            f"{D}CutFlag", f"{D}CutFlag && {D}_match_gLLP && "
                            + f"(200 < {D}_match_gLLP_decay_r) && "
                            + f"({D}_match_gLLP_decay_r < 800) && "
                            + f"(abs({D}_match_gLLP_decay_z) < 700)"
                        )
                    elif "r3" in key:
                        continue

                    if " n-1 " in args.lower():
                        for var in list(variables.keys()):
                            if var not in list(rdf.GetColumnNames()):
                                continue
                            else:
                                h = rdf.Histo1D((f"acceptance_{var}_{key}",f"acceptance_{var}_{key}",variables[var][0], variables[var][1], variables[var][2]),f"{var}")
                                h_list.append(h)

                # **** #
                # Trigger selections (HLT may be wrong in MC?)
                if "HLT" in cut:
                    # rdf = rdf.Redefine("evtCutFlag", "evtCutFlag && HLTDecision[566]")
                    if "mc" in key and "hlt" not in FN_MC:
                        continue
                    elif "r3" in key and "hlt" not in FN_R3:
                        # 566 = CscCluster_Loose
                        # 569 = HLT_L1CSCCluster_DTCluster50
                        rdf = rdf.Redefine("evtCutFlag", "evtCutFlag && HLTDecision[566]")
                    else:
                        continue

                    if " n-1 " in args.lower():
                        for var in list(variables.keys()):
                            if var not in list(rdf.GetColumnNames()):
                                continue
                            else:
                                h = rdf.Histo1D((f"HLT_{var}_{key}",f"HLT_{var}_{key}",variables[var][0], variables[var][1], variables[var][2]),f"{var}")
                                h_list.append(h)

                if "L1" in cut: # First passes, second fails
                    #fmt: off
                    # At least 1 IT cluster needs to pass the L1-turn on
                    # rdf = rdf.Redefine(f"evtCutFlag", f"evtCutFlag && {C}0CutFlag && ( "
                    # rdf = rdf.Define(f"passL1", f"{C}0CutFlag && ( "
                    #     + f"((100 < {C}R) && ({C}R < 275) && (580 < abs({C}Z)) && (abs({C}Z) < 632) && (500 <= {C}Size)) || "  # ME 11
                    #     + f"((275 < {C}R) && ({C}R < 465) && (668 < abs({C}Z)) && (abs({C}Z) < 724) && (200 <= {C}Size)) || "  # ME 12
                    #     + f"((505 < {C}R) && ({C}R < 700) && (668 < abs({C}Z)) && (abs({C}Z) < 724) && (200 <= {C}Size)) || "  # ME 13
                    #     #
                    #     + f"((139 < {C}R) && ({C}R < 345) && (789 < abs({C}Z)) && (abs({C}Z) < 850) && (500 <= {C}Size)) || "  # ME 21
                    #     + f"((357 < {C}R) && ({C}R < 700) && (791 < abs({C}Z)) && (abs({C}Z) < 850) && (200 <= {C}Size)) || "  # ME 22
                    #     #
                    #     + f"((160 < {C}R) && ({C}R < 345) && (915 < abs({C}Z)) && (abs({C}Z) < 970) && (500 <= {C}Size)) || "  # ME 31
                    #     + f"((357 < {C}R) && ({C}R < 700) && (911 < abs({C}Z)) && (abs({C}Z) < 970) && (200 <= {C}Size)) || "  # ME 32
                    #     #
                    #     + f"((178 < {C}R) && ({C}R < 345) && (1002 < abs({C}Z)) && (abs({C}Z) < 1063) && (500 <= {C}Size)) || "  # ME 41
                    #     + f"((357 < {C}R) && ({C}R < 700) && (1002 < abs({C}Z)) && (abs({C}Z) < 1063) && (200 <= {C}Size)) )"  # ME 42
                    # )
                    # rdf = rdf.Redefine(f"evtCutFlag", f"evtCutFlag && (reduce(passL1.begin(), passL1.end()) > 0)")
                    rdf = rdf.Redefine(f"evtCutFlag",
                        f"auto cscIT = ({MIN_CSC_TIME} < {C}TimeWeighted)"
                        f" && ({C}TimeWeighted < {MAX_CSC_TIME})"
                        f" && ({C}TimeSpreadWeightedAll < {MAX_CSC_TSPREAD});"
                        f"auto aboveL1 = cscIT && ("
                        f"((100 < {C}R) && ({C}R < 275) && (580 < abs({C}Z)) && (abs({C}Z) < 632) && (500 <= {C}Size))"  # ME 11
                        f" || ((275 < {C}R) && ({C}R < 465) && (668 < abs({C}Z)) && (abs({C}Z) < 724) && (200 <= {C}Size))"  # ME 12
                        f" || ((505 < {C}R) && ({C}R < 700) && (668 < abs({C}Z)) && (abs({C}Z) < 724) && (200 <= {C}Size))"  # ME 13
                        f" || ((139 < {C}R) && ({C}R < 345) && (789 < abs({C}Z)) && (abs({C}Z) < 850) && (500 <= {C}Size))"  # ME 21
                        f" || ((357 < {C}R) && ({C}R < 700) && (791 < abs({C}Z)) && (abs({C}Z) < 850) && (200 <= {C}Size))"  # ME 22
                        f" || ((160 < {C}R) && ({C}R < 345) && (915 < abs({C}Z)) && (abs({C}Z) < 970) && (500 <= {C}Size))"  # ME 31
                        f" || ((357 < {C}R) && ({C}R < 700) && (911 < abs({C}Z)) && (abs({C}Z) < 970) && (200 <= {C}Size))"  # ME 32
                        f" || ((178 < {C}R) && ({C}R < 345) && (1002 < abs({C}Z)) && (abs({C}Z) < 1063) && (500 <= {C}Size))"  # ME 41
                        f" || ((357 < {C}R) && ({C}R < 700) && (1002 < abs({C}Z)) && (abs({C}Z) < 1063) && (200 <= {C}Size))"  # ME 42
                        f" );"
                        f"return evtCutFlag && (reduce(aboveL1.begin(), aboveL1.end()) > 0)")

                    # The 1st cluster (largest cluster) needs to pass the L1-turn on
                    # rdf = rdf.Redefine(f"{C}0CutFlag", f"{C}0CutFlag && ( "
                    #     + f"((100 < {C}R) && ({C}R < 275) && (580 < abs({C}Z)) && (abs({C}Z) < 632) && (500 <= {C}Size)) || "  # ME 11
                    #     + f"((275 < {C}R) && ({C}R < 465) && (668 < abs({C}Z)) && (abs({C}Z) < 724) && (200 <= {C}Size)) || "  # ME 12
                    #     + f"((505 < {C}R) && ({C}R < 700) && (668 < abs({C}Z)) && (abs({C}Z) < 724) && (200 <= {C}Size)) || "  # ME 13
                    #     #
                    #     + f"((139 < {C}R) && ({C}R < 345) && (789 < abs({C}Z)) && (abs({C}Z) < 850) && (500 <= {C}Size)) || "  # ME 21
                    #     + f"((357 < {C}R) && ({C}R < 700) && (791 < abs({C}Z)) && (abs({C}Z) < 850) && (200 <= {C}Size)) || "  # ME 22
                    #     #
                    #     + f"((160 < {C}R) && ({C}R < 345) && (915 < abs({C}Z)) && (abs({C}Z) < 970) && (500 <= {C}Size)) || "  # ME 31
                    #     + f"((357 < {C}R) && ({C}R < 700) && (911 < abs({C}Z)) && (abs({C}Z) < 970) && (200 <= {C}Size)) || "  # ME 32
                    #     #
                    #     + f"((178 < {C}R) && ({C}R < 345) && (1002 < abs({C}Z)) && (abs({C}Z) < 1063) && (500 <= {C}Size)) || "  # ME 41
                    #     + f"((357 < {C}R) && ({C}R < 700) && (1002 < abs({C}Z)) && (abs({C}Z) < 1063) && (200 <= {C}Size)) )"  # ME 42
                    # )

                    # # The 2nd cluster does not need to pass the L1 turn-on
                    # rdf = rdf.Redefine(f"{C}1CutFlag", f"{C}1CutFlag && !( "
                    #     + f"((100 < {C}R) && ({C}R < 275) && (580 < abs({C}Z)) && (abs({C}Z) < 632) && (500 <= {C}Size)) || "  # ME 11
                    #     + f"((275 < {C}R) && ({C}R < 465) && (668 < abs({C}Z)) && (abs({C}Z) < 724) && (200 <= {C}Size)) || "  # ME 12
                    #     + f"((505 < {C}R) && ({C}R < 700) && (668 < abs({C}Z)) && (abs({C}Z) < 724) && (200 <= {C}Size)) || "  # ME 13
                    #     #
                    #     + f"((139 < {C}R) && ({C}R < 345) && (789 < abs({C}Z)) && (abs({C}Z) < 850) && (500 <= {C}Size)) || "  # ME 21
                    #     + f"((357 < {C}R) && ({C}R < 700) && (791 < abs({C}Z)) && (abs({C}Z) < 850) && (200 <= {C}Size)) || "  # ME 22
                    #     #
                    #     + f"((160 < {C}R) && ({C}R < 345) && (915 < abs({C}Z)) && (abs({C}Z) < 970) && (500 <= {C}Size)) || "  # ME 31
                    #     + f"((357 < {C}R) && ({C}R < 700) && (911 < abs({C}Z)) && (abs({C}Z) < 970) && (200 <= {C}Size)) || "  # ME 32
                    #     #
                    #     + f"((178 < {C}R) && ({C}R < 345) && (1002 < abs({C}Z)) && (abs({C}Z) < 1063) && (500 <= {C}Size)) || "  # ME 41
                    #     + f"((357 < {C}R) && ({C}R < 700) && (1002 < abs({C}Z)) && (abs({C}Z) < 1063) && (200 <= {C}Size)) )"  # ME 42
                    # )
                    #fmt: on


                    if " n-1 " in args.lower():
                        for var in list(variables.keys()):
                            if var not in list(rdf.GetColumnNames()):
                                continue
                            else:
                                h = rdf.Histo1D((f"L1_{var}_{key}",f"L1_{var}_{key}",variables[var][0], variables[var][1], variables[var][2]),f"{var}")
                                h_list.append(h)
                                



                if "CSC0 largest" in cut:
                    rdf = rdf.Redefine(f"{C}0CutFlag", f"{C}0CutFlag && ( {C}0Size == Max({C}Size*{C}0Flag) )")
                    rdf = rdf.Redefine(f"{C}1CutFlag", f"{C}1CutFlag && !{C}0CutFlag")

                    if " n-1 " in args.lower():
                        for var in list(variables.keys()):
                            if var not in list(rdf.GetColumnNames()):
                                continue
                            else:
                                h = rdf.Histo1D((f"CSC0_largest_{var}_{key}",f"CSC0_largest_{var}_{key}",variables[var][0], variables[var][1], variables[var][2]),f"{var}")
                                h_list.append(h)

                # if "CSC1 not CSC0" in cut:
                #     rdf = rdf.Redefine(f"{C}1CutFlag", f"{C}0CutFlag && ( {C}0Size == Max({C}Size*{C}0CutFlag) )")



                # **** #
                # Missing Et requirements & categorization
                if "MET" in cut:
                    rdf = rdf.Redefine("evtCutFlag", f"evtCutFlag && (met < 200)")


                    if " n-1 " in args.lower():
                        for var in list(variables.keys()):
                            if var not in list(rdf.GetColumnNames()):
                                continue
                            else:
                                h = rdf.Histo1D((f"MET_{var}_{key}",f"MET_{var}_{key}",variables[var][0], variables[var][1], variables[var][2]),f"{var}")
                                h_list.append(h)

                    
                if "low MET" in cut:
                    rdf = rdf.Redefine("evtCutFlag", f"evtCutFlag && (met < {LOW_MET_CUTOFF})")

                    if " n-1 " in args.lower():
                        for var in list(variables.keys()):
                            if var not in list(rdf.GetColumnNames()):
                                continue
                            else:
                                h = rdf.Histo1D((f"low_MET_{var}_{key}",f"low_MET_{var}_{key}",variables[var][0], variables[var][1], variables[var][2]),f"{var}")
                                h_list.append(h)

                    
                if "high MET" in cut:
                    rdf = rdf.Redefine("evtCutFlag", f"evtCutFlag && (({HIGH_MET_CUTOFF} < met) && (met < 200))")

                    if " n-1 " in args.lower():
                        for var in list(variables.keys()):
                            if var not in list(rdf.GetColumnNames()):
                                continue
                            else:
                                h = rdf.Histo1D((f"high_MET_{var}_{key}",f"high_MET_{var}_{key}",variables[var][0], variables[var][1], variables[var][2]),f"{var}")
                                h_list.append(h)

                    
                # **** #
                # Cluster time requirements
                if "CSC" in cut and "IT" in cut:
                    if "CSC0" in cut or ("0" not in cut and "1" not in cut):
                        rdf = rdf.Redefine(f"{C}0CutFlag", f"{C}0CutFlag && ( "
                            + f"({MIN_CSC_TIME} < {C}TimeWeighted) && "
                            + f"({C}TimeWeighted < {MAX_CSC_TIME}) && "
                            +f"({C}TimeSpreadWeightedAll < {MAX_CSC_TSPREAD}) )"
                        )
                        if " n-1 " in args.lower():
                            for var in list(variables.keys()):
                                if var not in list(rdf.GetColumnNames()):
                                    continue
                                else:
                                    h = rdf.Histo1D((f"CSC0_IT_{var}_{key}",f"CSC0_IT_{var}_{key}",variables[var][0], variables[var][1], variables[var][2]),f"{var}")
                                    h_list.append(h)

                    if "CSC1" in cut or ("0" not in cut and "1" not in cut):
                        rdf = rdf.Redefine(f"{C}1CutFlag", f"{C}1CutFlag && ( "
                            + f"({MIN_CSC_TIME} < {C}TimeWeighted) && "
                            + f"({C}TimeWeighted < {MAX_CSC_TIME}) && "
                            +f"({C}TimeSpreadWeightedAll < {MAX_CSC_TSPREAD}) )"
                        )
                        if " n-1 " in args.lower():
                            for var in list(variables.keys()):
                                if var not in list(rdf.GetColumnNames()):
                                    continue
                                else:
                                    h = rdf.Histo1D((f"CSC1_IT_{var}_{key}",f"CSC1_IT_{var}_{key}",variables[var][0], variables[var][1], variables[var][2]),f"{var}")
                                    h_list.append(h)


                if "CSC0 OOT" in cut:
                    rdf = rdf.Redefine(f"{C}0CutFlag", f"{C}0CutFlag && !( "
                        + f"({MIN_CSC_TIME} < {C}TimeWeighted) && "
                        + f"({C}TimeWeighted < {MAX_CSC_TIME}) && "
                        +f"({C}TimeSpreadWeightedAll < {MAX_CSC_TSPREAD}) )"
                    )

                    if " n-1 " in args.lower():
                        for var in list(variables.keys()):
                            if var not in list(rdf.GetColumnNames()):
                                continue
                            else:
                                h = rdf.Histo1D((f"CSC0_OOT_{var}_{key}",f"CSC0_OOT_{var}_{key}",variables[var][0], variables[var][1], variables[var][2]),f"{var}")
                                h_list.append(h)

                if "CSC1 OOT" in cut:
                    rdf = rdf.Redefine(f"{C}1CutFlag", f"{C}1CutFlag && !( "
                        + f"({MIN_CSC_TIME} < {C}TimeWeighted) && "
                        + f"({C}TimeWeighted < {MAX_CSC_TIME}) && "
                        +f"({C}TimeSpreadWeightedAll < {MAX_CSC_TSPREAD}) )"
                    )

                    if " n-1 " in args.lower():
                        for var in list(variables.keys()):
                            if var not in list(rdf.GetColumnNames()):
                                continue
                            else:
                                h = rdf.Histo1D((f"CSC1_OOT_{var}_{key}",f"CSC1_OOT_{var}_{key}",variables[var][0], variables[var][1], variables[var][2]),f"{var}")
                                h_list.append(h)


                if "DT IT" in cut:
                    rdf = rdf.Redefine(f"{D}CutFlag", f"{D}CutFlag && ( "
                        + f"(abs({D}_match_RPCBx_dPhi0p5) <= {MAX_RPC_BX}) && "
                        + f"({D}_match_RPChits_dPhi0p5 >= {MIN_RPC_HITS}) )"
                    )

                    if " n-1 " in args.lower():
                        for var in list(variables.keys()):
                            if var not in list(rdf.GetColumnNames()):
                                continue
                            else:
                                h = rdf.Histo1D((f"DT_IT_{var}_{key}",f"DT_IT_{var}_{key}",variables[var][0], variables[var][1], variables[var][2]),f"{var}")
                                h_list.append(h)

                elif "DT OOT" in cut:
                    rdf = rdf.Redefine(f"{D}CutFlag", f"{D}CutFlag && !( "
                        + f"(abs({D}_match_RPCBx_dPhi0p5) <= {MAX_RPC_BX}) && "
                        + f"({D}_match_RPChits_dPhi0p5 >= {MIN_RPC_HITS}) )"
                    )

                    if " n-1 " in args.lower():
                        for var in list(variables.keys()):
                            if var not in list(rdf.GetColumnNames()):
                                continue
                            else:
                                h = rdf.Histo1D((f"DT_OOT_{var}_{key}",f"DT_OOT_{var}_{key}",variables[var][0], variables[var][1], variables[var][2]),f"{var}")
                                h_list.append(h)


                # **** #
                # Station requirements from L1 trigger
                if "ME1" in cut: # and OPT_CUT != "ME1":
                    # The 1st cluster (largest) needs to pass the ME1=0 requirement from the HLT
                    if "CSC0" in cut or "CSC" not in cut:
                        rdf = rdf.Redefine(f"{C}0CutFlag", f"{C}0CutFlag && ( {C}NHitME1 <= 0 ) ")
                        # rdf = rdf.Redefine(f"{C}0CutFlag", f"{C}0CutFlag && ( {C}NHitME1 <= {MAX_ME1} ) ")
                    # The 2nd cluster does not have the requirement from the HLT
                    if "CSC1" in cut or "CSC" not in cut:
                        rdf = rdf.Redefine(f"{C}1CutFlag", f"{C}1CutFlag && ( {C}NHitME1 <= {MAX_ME1} ) ")
                    # rdf = rdf.Redefine(f"{C}1CutFlag", f"{C}1CutFlag && ( "+
                    #                 f"({C}NRechitChamberPlus11 <= {MAX_ME1}) &&"+
                    #                 f"({C}NRechitChamberPlus12 <= {MAX_ME1}) &&"+
                    #                 f"({C}NRechitChamberMinus11 <= {MAX_ME1}) &&"+
                    #                 f"({C}NRechitChamberMinus12 <= {MAX_ME1}) )")

                    if " n-1 " in args.lower():
                        for var in list(variables.keys()):
                            if var not in list(rdf.GetColumnNames()):
                                continue
                            else:
                                h = rdf.Histo1D((f"ME1_{var}_{key}",f"ME1_{var}_{key}",variables[var][0], variables[var][1], variables[var][2]),f"{var}")
                                h_list.append(h)

                    
                if "MB1" in cut: # and OPT_CUT != "MB1":
                    # No HLT MB1 requirement when using HLT='CscCluster_Loose'
                    rdf = rdf.Redefine(f"{D}CutFlag", f"{D}CutFlag && ( {D}NHitStation1 <= {MAX_MB1} )")

                    if " n-1 " in args.lower():
                        for var in list(variables.keys()):
                            if var not in list(rdf.GetColumnNames()):
                                continue
                            else:
                                h = rdf.Histo1D((f"MB1_{var}_{key}",f"MB1_{var}_{key}",variables[var][0], variables[var][1], variables[var][2]),f"{var}")
                                h_list.append(h)

                if "DT stn" in cut:
                    rdf = rdf.Redefine(f"{D}CutFlag", f"{D}CutFlag && ( "
                        + f"({D}NStation10 < 3) && "
                        + f"!(({D}NStation10 == 2) && ({D}MaxStation == 4)) )"
                    )

                    if " n-1 " in args.lower():
                        for var in list(variables.keys()):
                            if var not in list(rdf.GetColumnNames()):
                                continue
                            else:
                                h = rdf.Histo1D((f"DT_stn_{var}_{key}",f"DT_stn_{var}_{key}",variables[var][0], variables[var][1], variables[var][2]),f"{var}")
                                h_list.append(h)


                # **** #
                if "CSC jet req" in cut:
                    rdf = rdf.Redefine("evtCutFlag",
                    f"auto passJetMatch = ({C}0Flag || {C}1Flag) && ({C}JetVetoPt >= {MAX_CSC_JET});"
                    f"return evtFlag && (reduce(passJetMatch.begin(), passJetMatch.end()) > 0)")

                    if " n-1 " in args.lower():
                        for var in list(variables.keys()):
                            if var not in list(rdf.GetColumnNames()):
                                continue
                            else:
                                h = rdf.Histo1D((f"CSC_jet_req_{var}_{key}",f"CSC_jet_req_{var}_{key}",variables[var][0], variables[var][1], variables[var][2]),f"{var}")
                                h_list.append(h)

                if "CSC jet veto" in cut: # and OPT_CUT != "CSC jet veto":
                    if "$<$" in cut:
                        MAX_CSC_JET = float(cut.split(" ")[-1])
                    rdf = rdf.Redefine(f"{C}0CutFlag", f"{C}0CutFlag && ({C}JetVetoPt < {MAX_CSC_JET})")
                    rdf = rdf.Redefine(f"{C}1CutFlag", f"{C}1CutFlag && ({C}JetVetoPt < {MAX_CSC_JET})")

                    if " n-1 " in args.lower():
                        for var in list(variables.keys()):
                            if var not in list(rdf.GetColumnNames()):
                                continue
                            else:
                                h = rdf.Histo1D((f"CSC_jet_veto_{var}_{key}",f"CSC_jet_veto_{var}_{key}",variables[var][0], variables[var][1], variables[var][2]),f"{var}")
                                h_list.append(h)

                if "DT jet veto" in cut: # and OPT_CUT != "DT jet veto":
                    rdf = rdf.Redefine(f"{D}CutFlag", f"{D}CutFlag && ({D}JetVetoPt < {MAX_DT_JET})")

                    if " n-1 " in args.lower():
                        for var in list(variables.keys()):
                            if var not in list(rdf.GetColumnNames()):
                                continue
                            else:
                                h = rdf.Histo1D((f"DT_jet_veto_{var}_{key}",f"DT_jet_veto_{var}_{key}",variables[var][0], variables[var][1], variables[var][2]),f"{var}")
                                h_list.append(h)

                    
                if "CSC muon veto" in cut: # and OPT_CUT != "CSC muon veto":
                    # rdf = rdf.Redefine(f"{C}0CutFlag", f"{C}0CutFlag && ({C}MuonVetoPt < {abs(MAX_CSC_MUON)})")
                    # rdf = rdf.Redefine(f"{C}1CutFlag", f"{C}1CutFlag && ({C}MuonVetoPt < {abs(MAX_CSC_MUON)})")
                    rdf = rdf.Redefine(f"{C}0CutFlag", f"{C}0CutFlag && "
                        + f"!( !({C}MuonVetoGlobal == 0) && !({C}MuonVetoPt < {MAX_CSC_MUON}) )"
                    )
                    rdf = rdf.Redefine(f"{C}1CutFlag", f"{C}1CutFlag && "
                        + f"!( !({C}MuonVetoGlobal == 0) && !({C}MuonVetoPt < {MAX_CSC_MUON}) )"
                    )

                    if " n-1 " in args.lower():
                        for var in list(variables.keys()):
                            if var not in list(rdf.GetColumnNames()):
                                continue
                            else:
                                h = rdf.Histo1D((f"CSC_muon_veto_{var}_{key}",f"CSC_muon_veto_{var}_{key}",variables[var][0], variables[var][1], variables[var][2]),f"{var}")
                                h_list.append(h)

                if "DT muon veto" in cut: # and OPT_CUT != "DT muon veto":
                    # rdf = rdf.Redefine(f"{D}CutFlag", f"{D}CutFlag && ({D}MuonVetoPt < {abs(MAX_DT_MUON)})")
                    rdf = rdf.Redefine(f"{D}CutFlag", f"{D}CutFlag && "
                        + f"!( !({D}MuonVetoLooseId == 0) && !({D}MuonVetoPt < {MAX_DT_MUON}) )"
                    )

                    if " n-1 " in args.lower():
                        for var in list(variables.keys()):
                            if var not in list(rdf.GetColumnNames()):
                                continue
                            else:
                                h = rdf.Histo1D((f"DT_muon_veto_{var}_{key}",f"DT_muon_veto_{var}_{key}",variables[var][0], variables[var][1], variables[var][2]),f"{var}")
                                h_list.append(h)

                if "halo veto" in cut: # and OPT_CUT != "halo veto":
                    rdf = rdf.Redefine(f"{D}CutFlag", f"{D}CutFlag && "
                        + f"({HALO_CUTOFF} < abs({D}Phi)) && "
                        + f"(abs({D}Phi) < {PI} - {HALO_CUTOFF})"
                    )

                    if " n-1 " in args.lower():
                        for var in list(variables.keys()):
                            if var not in list(rdf.GetColumnNames()):
                                continue
                            else:
                                h = rdf.Histo1D((f"halo_veto_{var}_{key}",f"halo_veto_{var}_{key}",variables[var][0], variables[var][1], variables[var][2]),f"{var}")
                                h_list.append(h)

                # **** #
                if "BDT" in cut:
                    raise NotImplementedError("BDT")

                if "DNN $>$" in cut: 
                    rdf = rdf.Redefine(f"{C}0CutFlag", f"{C}0CutFlag && "
                        + f"( Take({C}DNN_{DNN_VERSION},nCscRechitClusters) > {MIN_CSC_DNN} )"
                    )
                    rdf = rdf.Redefine(f"{C}1CutFlag", f"{C}1CutFlag && "
                        + f"( Take({C}DNN_{DNN_VERSION},nCscRechitClusters) > {MIN_CSC_DNN} )"
                    )
                    

                    if " n-1 " in args.lower():
                        for var in list(variables.keys()):
                            if var not in list(rdf.GetColumnNames()):
                                continue
                            else:
                                h = rdf.Histo1D((f"DNN_$>$_{var}_{key}",f"DNN_$>$_{var}_{key}",variables[var][0], variables[var][1], variables[var][2]),f"{var}")
                                h_list.append(h)

                if "DNN $<$" in cut: 
                    rdf = rdf.Redefine("evtCutFlag",
                    f"auto invertDNN = ({C}0Flag || {C}1Flag) && (Take({C}DNN_{DNN_VERSION},nCscRechitClusters) <= {MIN_CSC_DNN});"
                    f"return evtFlag && (reduce(invertDNN.begin(), invertDNN.end()) > 0)")

                    if " n-1 " in args.lower():
                        for var in list(variables.keys()):
                            if var not in list(rdf.GetColumnNames()):
                                continue
                            else:
                                h = rdf.Histo1D((f"DNN_$<$_{var}_{key}",f"DNN_$<$_{var}_{key}",variables[var][0], variables[var][1], variables[var][2]),f"{var}")
                                h_list.append(h)

                # **** #
                if "0 DT" in cut:
                    # rdf = rdf.Redefine("evtCutFlag", f"evtCutFlag && (reduce({D}Flag.begin(), {D}Flag.end()) == 0)")
                    rdf = rdf.Redefine("evtCutFlag", "evtCutFlag && (nDtRechitClusters == 0)")

                    if " n-1 " in args.lower():
                        for var in list(variables.keys()):
                            if var not in list(rdf.GetColumnNames()):
                                continue
                            else:
                                h = rdf.Histo1D((f"0_DT_{var}_{key}",f"0_DT_{var}_{key}",variables[var][0], variables[var][1], variables[var][2]),f"{var}")
                                h_list.append(h)

                if "2 CSC" in cut:
                    rdf = rdf.Redefine("evtCutFlag", "evtCutFlag && (nCscRechitClusters == 2)")

                    if " n-1 " in args.lower():
                        for var in list(variables.keys()):
                            if var not in list(rdf.GetColumnNames()):
                                continue
                            else:
                                h = rdf.Histo1D((f"2_CSC_{var}_{key}",f"2_CSC_{var}_{key}",variables[var][0], variables[var][1], variables[var][2]),f"{var}")
                                h_list.append(h)

        
                if "1 CSC-CSC" in cut:
                    if "Exactly" in cut:
                        rdf = rdf.Redefine("evtCutFlag", "evtCutFlag && (nCscRechitClusters == 2)")

                        if " n-1 " in args.lower():
                            for var in list(variables.keys()):
                                if var not in list(rdf.GetColumnNames()):
                                    continue
                                else:
                                    h = rdf.Histo1D((f"Exactly_1_CSC-CSC_{var}_{key}",f"Exactly_1_CSC-CSC_{var}_{key}",variables[var][0], variables[var][1], variables[var][2]),f"{var}")
                                    h_list.append(h)

                    # rdf = rdf.Redefine("evtCutFlag", f"evtCutFlag && evtFlag && "+
                    #                    f"(reduce({C}0Flag.begin(), {C}0Flag.end()) == 1) && "+
                    #                    f"(reduce({C}1Flag.begin(), {C}1Flag.end()) == 1)")
                    rdf = rdf.Redefine(f"{C}0Flag", f"{C}0Flag && ( {C}0Size == Max({C}Size*{C}0Flag) )")
                    rdf = rdf.Redefine(f"{C}1Flag", f"{C}1Flag && ( {C}1Size == Max({C}Size*({C}1Flag && !{C}0Flag)) )")

                    rdf = rdf.Redefine("evtCutFlag", f"evtCutFlag && evtFlag")
                    rdf = rdf.Redefine(f"{C}0CutFlag", f"{C}0CutFlag && {C}0CutFlag")
                    rdf = rdf.Redefine(f"{C}1CutFlag", f"{C}1CutFlag && {C}1CutFlag")
                    # rdf = rdf.Redefine(f"{D}Flag", f"{D}Flag && ( {D}Size == Max({D}Size*{D}Flag) )")

                    # Apply our cluster level selections to relevant columns
                    for col in COLUMNS_OUT:
                        if C in col[:len(C)]:
                            ocol0 = col.replace("RechitCluster", "0")
                            ocol1 = col.replace("RechitCluster", "1")
                            rdf = rdf.Define(ocol0, f"{col}[{C}0Flag][0]")
                            rdf = rdf.Define(ocol1, f"{col}[{C}1Flag][0]")
                            if ocol0 not in columns_out:
                                columns_out.append(ocol0)
                            if ocol1 not in columns_out:
                                columns_out.append(ocol1)
                        elif D in col[:len(D)]:
                            ocol = col.replace("RechitCluster", "")
                            rdf = rdf.Define(ocol, f"{col}[{D}Flag][0]")
                            if ocol not in columns_out:
                                columns_out.append(ocol)
                        else:
                            if col not in columns_out:
                                columns_out.append(col)

                    rdf = rdf.Define("csc0DNN_bkgMC", f"Take({C}DNN_bkgMC,nCscRechitClusters)[{C}0Flag][0]")
                    rdf = rdf.Define(
                        "csc0DNN_bkgMC_plusBeamHalo", f"Take({C}DNN_bkgMC_plusBeamHalo,nCscRechitClusters)[{C}0Flag][0]"
                    )
                    rdf = rdf.Define("csc0DNN_bkgOOTData", f"Take({C}DNN_bkgOOTData,nCscRechitClusters)[{C}0Flag][0]")

                    rdf = rdf.Define("csc1DNN_bkgMC", f"Take({C}DNN_bkgMC,nCscRechitClusters)[{C}1Flag][0]")
                    rdf = rdf.Define(
                        "csc1DNN_bkgMC_plusBeamHalo", f"Take({C}DNN_bkgMC_plusBeamHalo,nCscRechitClusters)[{C}1Flag][0]"
                    )
                    rdf = rdf.Define("csc1DNN_bkgOOTData", f"Take({C}DNN_bkgOOTData,nCscRechitClusters)[{C}1Flag][0]")

                    rdf = rdf.Define("csc0CTau", f"gLLP_ctau[{C}_match_gLLP_index[{C}0Flag][0]]")
                    rdf = rdf.Define("csc1CTau", f"gLLP_ctau[{C}_match_gLLP_index[{C}1Flag][0]]")
                    rdf = rdf.Define("tag_dEta", "abs(csc0Eta - csc1Eta)")
                    rdf = rdf.Define("tag_dPhi", f"abs( abs(csc0Phi - csc1Phi) - (abs(csc0Phi - csc1Phi) > {PI} ? 2*{PI} : 0) )")
                    rdf = rdf.Define("tag_dR", "sqrt(tag_dEta*tag_dEta + tag_dPhi*tag_dPhi)")
                    columns_out.extend(
                        [
                            "tag_dEta", 
                            "tag_dPhi", 
                            "tag_dR", 
                            "csc0CTau", 
                            "csc1CTau",
                            "csc0DNN_bkgMC",
                            "csc0DNN_bkgMC_plusBeamHalo",
                            "csc0DNN_bkgOOTData",
                            "csc1DNN_bkgMC",
                            "csc1DNN_bkgMC_plusBeamHalo",
                            "csc1DNN_bkgOOTData",
                        ]
                    )

                    if " n-1 " in args.lower():
                        for var in list(variables.keys()):
                            if var not in list(rdf.GetColumnNames()):
                                continue
                            else:
                                h = rdf.Histo1D((f"1_CSC-CSC_{var}_{key}",f"1_CSC_CSC_{var}_{key}",variables[var][0], variables[var][1], variables[var][2]),f"{var}")
                                h_list.append(h)

                # **** #
                if "dR" in cut:
                    raise NotImplementedError("dR")

                if "dEta" in cut:
                    raise NotImplementedError("dEta")

                if "dEta" in cut:
                    # if OPT_CUT != "min dEta":
                    rdf = rdf.Redefine("evtCutFlag", f"evtCutFlag && (tag_dEta > {MIN_DETA})")
                    # if OPT_CUT != "max dEta":
                    rdf = rdf.Redefine("evtCutFlag", f"evtCutFlag && (tag_dEta < {MAX_DETA})")

                    if " n-1 " in args.lower():
                        for var in list(variables.keys()):
                            if var not in list(rdf.GetColumnNames()):
                                continue
                            else:
                                h = rdf.Histo1D((f"dEta_{var}_{key}",f"dEta_{var}_{key}",variables[var][0], variables[var][1], variables[var][2]),f"{var}")
                                h_list.append(h)

                if "dPhi" in cut: # and # OPT_CUT != "min dPhi":
                    rdf = rdf.Redefine("evtCutFlag", f"evtCutFlag && (tag_dPhi > {MIN_DPHI})")

                    if " n-1 " in args.lower():
                        for var in list(variables.keys()):
                            if var not in list(rdf.GetColumnNames()):
                                continue
                            else:
                                h = rdf.Histo1D((f"dPhi_{var}_{key}",f"dPhi_{var}_{key}",variables[var][0], variables[var][1], variables[var][2]),f"{var}")
                                h_list.append(h)

                # **** #
                # Propagate to cumulative flags
                rdf = rdf.Redefine(f"{C}0Flag", f"{C}0Flag && {C}0CutFlag")
                rdf = rdf.Redefine(f"{C}1Flag", f"{C}1Flag && {C}1CutFlag")
                rdf = rdf.Redefine(f"{D}Flag", f"{D}Flag && {D}CutFlag")

                # print(
                #     rdf.Filter("evtFlag").Sum(f"{C}0Flag").GetValue(),
                #     rdf.Filter("evtFlag").Sum(f"{C}1Flag").GetValue(),
                #     rdf.Filter("evtFlag").Sum(f"{D}Flag").GetValue()
                # , flush=True)

                rdf = rdf.Redefine("evtCutFlag", 
                    f"auto {C}01CutFlag = {C}0CutFlag || {C}1CutFlag;"
                    + f"return evtCutFlag && ( "
                    + f"(reduce({C}0CutFlag.begin(), {C}0CutFlag.end()) > 0) && "
                    + f"(reduce({C}1CutFlag.begin(), {C}1CutFlag.end()) > 0) && "
                    + f"(reduce({C}01CutFlag.begin(), {C}01CutFlag.end()) > 1) );"
                )

                rdf = rdf.Redefine("evtFlag", 
                    f"auto {C}01Flag = {C}0Flag || {C}1Flag;"
                    + f"return evtFlag && evtCutFlag && ( "
                    + f"(reduce({C}0Flag.begin(), {C}0Flag.end()) > 0) && "
                    + f"(reduce({C}1Flag.begin(), {C}1Flag.end()) > 0) && "
                    + f"(reduce({C}01Flag.begin(), {C}01Flag.end()) > 1) );"
                )

                # rdf = rdf.Redefine("evtCutFlag", "evtCutFlag && ("
                #     + f"(reduce({C}0CutFlag.begin(), {C}0CutFlag.end()) > 0) && "
                #     + f"(reduce({C}1CutFlag.begin(), {C}1CutFlag.end()) > 0) )"
                # )

                # rdf = rdf.Redefine("evtFlag", "evtFlag && evtCutFlag && ("
                #     + f"(reduce({C}0Flag.begin(), {C}0Flag.end()) > 0) && "
                #     + f"(reduce({C}1Flag.begin(), {C}1Flag.end()) > 0) )"
                # )

                # rdf = rdf.Redefine(f"nC{C[1:]}s", f"reduce({C}Flag.begin(), {C}Flag.end())")
                # rdf = rdf.Redefine(f"nD{D[1:]}s", f"reduce({D}Flag.begin(), {D}Flag.end())")

                # **** #
                if PRINT_CUTFLOW:
                    ec,cc0,cc1,dc = rdf.Filter("evtFlag").Sum("weight"), rdf.Filter("evtFlag").Sum(f"{C}0Flag"), rdf.Filter("evtFlag").Sum(f"{C}1Flag"), rdf.Filter("evtFlag").Sum(f"{D}Flag")
                    # ecc,ccc0,ccc1,dcc = rdf.Filter("evtCutFlag").Sum("weight"), rdf.Filter("evtCutFlag").Sum(f"{C}0CutFlag"), rdf.Filter("evtCutFlag").Sum(f"{C}1CutFlag"), rdf.Filter("evtCutFlag").Sum(f"{D}CutFlag")
                    ecc,ccc0,ccc1,dcc = rdf.Filter("evtCutFlag").Sum("weight"), rdf.Sum(f"{C}0CutFlag"), rdf.Sum(f"{C}1CutFlag"), rdf.Sum(f"{D}CutFlag")
                    
                    ec,cc0,cc1,dc = ec.GetValue(),cc0.GetValue(),cc1.GetValue(),dc.GetValue()
                    ecc,ccc0,ccc1,dcc = ecc.GetValue(),ccc0.GetValue(),ccc1.GetValue(),dcc.GetValue()

                    c0_cum_eff = f"{100*cc0/cc00:.2g}" if cc0 != cc00 else "--"
                    c0_cut_eff = f"{100*ccc0/cc00:.2g}" if ccc0 != cc00 else "--"
                    c1_cum_eff = f"{100*cc1/cc10:.2g}" if cc1 != cc10 else "--"
                    c1_cut_eff = f"{100*ccc1/cc10:.2g}" if ccc1 != cc10 else "--"
                    d_cum_eff = f"{100*dc/dc0:.2g}" if dc != dc0 else "--"
                    d_cut_eff = f"{100*dcc/dc0:.2g}" if dcc != dc0 else "--"

                    # print(f"    {cut.replace('_',' ')} & {ec:,.0f} & {100*ecc/ec0:.2f} & {100*ec/ec0:.2f} & "+
                    #     f"{cc0:,.0f} & {c0_cut_eff} & {c0_cum_eff} & "+
                    #     f"{cc1:,.0f} & {c1_cut_eff} & {c1_cum_eff} & "+
                    #     f"{dc:,.0f} & {d_cut_eff} & {d_cum_eff} \\\\", flush=True)
                    # print(f"    {cut.replace('_',' ')} & {ec:,.0f} & {100*ecc/ec0:.2g} & {100*ec/ec0:.2g} & "+
                    #     f"{cc0:,.0f} & {c0_cut_eff} & {c0_cum_eff} & "+
                    #     f"{cc1:,.0f} & {c1_cut_eff} & {c1_cum_eff} & "+
                    #     f"{dc:,.0f} & {d_cut_eff} & {d_cum_eff} \\\\", flush=True)
                    print(f"    {cut.replace('_',' ')} & {ec:,.2f} & {100*ec/ec0:.2g} & "+
                        f"{cc0:,.0f} & {c0_cum_eff} & "+
                        f"{cc1:,.0f} & {c1_cum_eff} & "+
                        f"{dc:,.0f} & {d_cum_eff} \\\\", flush=True)
                    
                    # Only filter (reset indices) after the MET cut when printing cutflow
                    # if any([_cut in cut for _cut in ("MET","acceptance","1 CSC-CSC")]):
                    #     print(r"    \hline", flush=True)
                    #     ec0, cc00, cc10, dc0 = ec, cc0, cc1, dc
                    
                    rdf = rdf.Filter("evtFlag")
                    rdf = rdf.Redefine(f"{C}0Size", f"{C}0Size * {C}0Flag")
                    rdf = rdf.Redefine(f"{C}1Size", f"{C}1Size * {C}1Flag")
                    rdf = rdf.Redefine(f"{D}Size", f"{D}Size * {D}Flag")
                else:
                    rdf = rdf.Filter("evtFlag")
                    rdf = rdf.Redefine(f"{C}0Size", f"{C}0Size * {C}0Flag")
                    rdf = rdf.Redefine(f"{C}1Size", f"{C}1Size * {C}1Flag")
                    rdf = rdf.Redefine(f"{D}Size", f"{D}Size * {D}Flag")


            # Apply the final filter (should be redundant) and update the dictionary with the new RDF 
            rdfs[key] = rdf.Filter("evtFlag")

            if PRINT_CUTFLOW:
                print(r"    \hline", flush=True)
                print(r"\end{tabular}", flush=True)
                print(r"\end{center}", flush=True)
                print("", flush=True)

        # **************** #
        if RAND:
            accepted_cut = False
            wmc = rdfs["mc"].Sum("weight").GetValue()
            wr3 = rdfs["r3"].Sum("weight").GetValue()
            score = wmc / (wr3**0.5) if wr3 else 0  # if b~1 then s2b explodes and falls into a false minimum
            # score = 2 * ((wmc + wr3) ** 0.5 - wr3**0.5)  # https://arxiv.org/pdf/hep-ph/0204326.pdf

            if OPT_CUT == "":
                # limit0 = limit
                score0, wmc0, wr30 = score, wmc, wr3
                val, val0 = 999, 999
            else:
                # cv_idx = np.argwhere(val == CUT_OPT_PARS[OPT_CUT]["values"])
                # cv0_idx = np.argwhere(val0 == CUT_OPT_PARS[OPT_CUT]["values"])
                cv_idx = np.searchsorted(CUT_OPT_PARS[OPT_CUT]["values"], val)
                cv0_idx = np.searchsorted(CUT_OPT_PARS[OPT_CUT]["values"], val0)
                # if wmc>0 and wr3>0 and limit <= limit0:# and wmc/wmc0 > 0.9:
                if wmc > 0 and wr3 > 0 and (score > score0 or (score==score0 and cv_idx<cv0_idx)):  # and wmc/wmc0 > 0.9:
                    accepted_cut = True
                    # CUT_VALUES[CUT_OPT_PARS[OPT_CUT]["cut"]] = val
                    CUT_VALUES[CUT_OPT_PARS[OPT_CUT]["cut"]] = (wmc*val + wmc0*val0)/(wmc+wmc0)
                else:
                    CUT_VALUES[CUT_OPT_PARS[OPT_CUT]["cut"]] = val0

            # print(f"{iopt:>3} | {"Y" if _c else "X"} | {OPT_CUT:>13} = {val:>6.2f} ({val0:>6.2f}) | {score:>3.0f} ({score0:>3.0f}), {wmc:>4.0f} ({wmc0:>4.0f}), {wr3:>5.0f} ({wr30:>5.0f})", flush=True)
            print(
                f"{iopt:>3} | {'Y' if accepted_cut else ' '} | {OPT_CUT:>13} = {val:>8.3f} ({val0:>8.3f}) | {score:>4.2f} ({score0:>4.2f}), {wmc:>4.0f} ({wmc0:>4.0f}), {wr3:>5.0f} ({wr30:>5.0f})"
            , flush=True)
            # print(f"{iopt:>3} | {"Y" if _c else "X"} | {OPT_CUT:>13} = {val:>6.2f} ({val0:>6.2f}) | {limit:>.2e} ({limit0:>.2e}), {wmc:>4.0f} ({wmc0:>4.0f}), {wr3:>5.0f} ({wr30:>5.0f})", flush=True)

            OPT_SCORES.append(
                [
                    OPT_CUT,
                    val,
                    score,
                    wmc,
                    wr3,
                    val0,
                    score0,
                    wmc0,
                    wr30,
                    {k: v for k, v in CUT_VALUES.items()},
                ]
            )
            # with open(
            #     f"rand_scores_cscdt{"OOT" if OOT else ""}_{CUTSET}_{MET_CATEGORY}.pkl",
            #     "wb",
            # ) as fout:
            #     pickle.dump(OPT_SCORES, fout)

            if accepted_cut:
                score0, wmc0, wr30 = score, wmc, wr3
                # limit0 = limit
            # score0 = 2*((wmc0*(2.16e-03**((iopt+1)/N_ITERATIONS))+wr30)**0.5 - wr30**0.5)

        # **************** #
        if (LOO or RAND) and (iopt%100 == 0 or iopt == N_ITERATIONS-1):
            print("", flush=True)
            print(f"Cut Boundaries: {wmc=:.2f}, {wr3=:.2f}, {score=:.2f}", flush=True)
            print(f"    {MIN_CSC_TIME=:.2f}", flush=True)
            print(f"    {MAX_CSC_TIME=:.2f}", flush=True)
            print(f"    {MAX_CSC_TSPREAD=:.2f}", flush=True)
            print(f"    {MAX_RPC_BX=:.0f}", flush=True)
            print(f"    {MIN_RPC_HITS=:.0f}", flush=True)
            print(f"    {MAX_CSC_JET=:.2f}", flush=True)
            print(f"    {MAX_DT_JET=:.2f}", flush=True)
            print(f"    {MAX_CSC_MUON=:.2f}", flush=True)
            print(f"    {MAX_DT_MUON=:.2f}", flush=True)
            print(f"    {MAX_ME1=:.0f}", flush=True)
            print(f"    {MAX_MB1=:.0f}", flush=True)
            print(f"    {HALO_CUTOFF=:.2f}", flush=True)
            print(f"    {MIN_DPHI=:.2f}", flush=True)
            print(f"    {MIN_DETA=:.2f}", flush=True)
            print(f"    {MAX_DETA=:.2f}", flush=True)
            print(f"    {MIN_CSC_DNN=:.3f}", flush=True)
            # print(f"    {MIN_DT_DNN=:.3f}", flush=True)
            print("", flush=True)


    # **************** #
    hname_pre = f"csccsc{'OOT' if OOT else ''}_{CUTSET}"
    if DNN_VERSION is not None and "DNN" in CUTSET:
        hname_pre += "DNN"
        # hname_pre += f"_{DNN_VERSION}"
    hname_pre += f"_{MET_CATEGORY}"
    if LOO or RAND:
        hname_pre += f"_LOO" if LOO else f"_RAND"
    if ITVal1:
        hname_pre += "_ITVal1"
    if ITVal2:
        hname_pre += "_ITVal2"
    hname_pre += f"_{YEAR}"

    print("Events out:", flush=True)
    hists = {}
    for key, rdf in rdfs.items():
        count, weight = rdf.Count(), rdf.Sum("weight")
        count, weight = count.GetValue(), weight.GetValue()
        if count == 0:
            print(f"  {key} is empty", flush=True)
            continue
        
        name = f"{key}_csccsc{'OOT' if OOT else ''}_{CUTSET}"
        if DNN_VERSION is not None and "DNN" in CUTSET:
            name += "DNN"
            # name += f"_{DNN_VERSION}"
        name += f"_{MET_CATEGORY}"

        if LOO or RAND:
            name += f"_LOO" if LOO else f"_RAND"
            if key == "mc":
                name = name.replace("cscdtOOT", "cscdt")
        
        if ITVal1:
            name += "_ITVal1"
        if ITVal2:
            name += "_ITVal2"

        name += f"_{YEAR}"
        print(f"  {key} = {count:,} ({weight:,.2f})", flush=True)
        rdf = rdf.Snapshot("MuonSystem_flat", f"data/processed/{name}_rdf.root", columns_out)
        rdfs[key] = rdf

        for xx in (
            "met", 
            "csc0Size", 
            "csc0R", 
            "csc0Eta", 
            "csc0Phi", 
            "csc1Size", 
            "csc1R", 
            "csc1Eta", 
            "csc1Phi", 
            "tag_dEta",
            "tag_dPhi",
            "tag_dR"
        ):
            hh = rdf.Histo1D(xx, "weight")
            hh.SetName(f"{key}_{xx}")
            hh.SetTitle(f"{key};{xx};count")
            if xx not in hists:
                hists[xx] = [hh]
            else:
                hists[xx].append(hh)
            canvas = rt.TCanvas("","",800,800)
            xmin, xmax, std = rdf.Min(xx).GetValue(), rdf.Max(xx).GetValue(), rdf.StdDev(xx).GetValue()
            nbins = int((xmax-xmin)/(2.718*std*count**(-1/3))) if std else 1
            hh = rdf.Histo1D((f"{key}_{xx}",f"{key};{xx};count",nbins,xmin,xmax),f"{xx}").GetValue()
            hh.SetMinimum(0)
            hh.Draw()
            canvas.Draw()
            canvas.Print(f"{OUT_DIR}/{name}_{xx}.png")
    for xx, hhs in hists.items():
        canvas = rt.TCanvas("","",800,800)
        # xmin, xmax, std = rdf.Min(xx).GetValue(), rdf.Max(xx).GetValue(), rdf.StdDev(xx).GetValue()
        # nbins = int((xmax-xmin)/(2.718*std*count**(-1/3))) if std else 1
        # hh = rdf.Histo1D((f"{key}_{xx}",f"{key};{xx};count",nbins,xmin,xmax),f"{xx}").GetValue()
        hhs = [h.GetValue() for h in hhs]
        # hmax = max([h.GetMaximum() / h.Integral() if h.Integral() else 0 for h in hhs])
        hmax = max([h.GetMaximum() for h in hhs])
        hmin = max([h.GetMinimum(0) for h in hhs])
        if hmax/hmin > 100:
            canvas.SetLogy()
        for ih, hh in enumerate(hhs):
            # if hmax < 1 and hh.Integral():
            #     hh.Scale(1/hh.Integral())
            hh.SetMinimum(hmin*0.9)
            hh.SetMaximum(hmax*1.1)
            # hh.GetXaxis().SetRangeUser(xmin, xmax)
            hh.SetLineColor(SCL[ih])
            hh.SetLineWidth(3)
            hh.Draw("hist same")
        canvas.Draw()
        canvas.Print(f"{OUT_DIR}/{hname_pre}_{xx}.png")

    print("", flush=True)

    
    if " n-1 " in args.lower():
        rt.gStyle.SetOptStat(0)
        for c in CUTS:
            c = c.replace(" ", "_")
            for var in list(variables.keys()):
                h_mc, h_data = None, None
                for h in h_list:
                    if((h.GetName().startswith(c + '_')) and ('_' + var + '_' in h.GetName()) and ('_mc' in h.GetName())):
                        h_mc = h
                        h_mc.SetLineColor(2)
                        h_mc.SetLineWidth(2)
                        h_mc.GetXaxis().SetTitle(var)
                        # include the overflow bin
                        h_mc.SetBinContent(h_mc.GetNbinsX(), h_mc.GetBinContent(h_mc.GetNbinsX()) + h_mc.GetBinContent(h_mc.GetNbinsX() + 1))
                    elif((h.GetName().startswith(c + '_')) and ('_' + var + '_' in h.GetName()) and ('_r3' in h.GetName())):
                        h_data = h
                        h_data.SetLineWidth(2)
                        # include the overflow bin
                        h_data.SetBinContent(h_data.GetNbinsX(), h_data.GetBinContent(h_data.GetNbinsX()) + h_data.GetBinContent(h_data.GetNbinsX() + 1))

                if((h_mc is not None) and (h_data is not None)):
                    h_mc.Scale(1./h_mc.Integral())
                    h_data.Scale(1./h_data.Integral())
                    canvas = rt.TCanvas("","",800,800)
                    canvas.cd()
                    h_mc.Draw()
                    h_data.Draw("same")
                    leg = rt.TLegend(.7,.8,.85,.9)
                    leg.SetBorderSize(0)
                    leg.SetFillColor(0)
                    leg.SetFillStyle(0)
                    leg.SetTextFont(42)
                    leg.SetTextSize(0.035)
                    leg.AddEntry(h_mc.GetPtr(),"Signal MC","l")
                    if OOT:
                        leg.AddEntry(h_data.GetPtr(),"OOT DATA","l")
                    else:
                        leg.AddEntry(h_data.GetPtr(),"IT DATA","l")
                    leg.Draw()
                    canvas.Draw()
                    c = c.replace('$','').replace('>', 'gt').replace('<','lt')
                    canvas.Print(f"{OUT_DIR}/N_1_CUT_{c}_VAR_{var}.png")

