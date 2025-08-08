import os
import numpy as np


class Config:
    def __init__(self):
        #cwd = os.getcwd()
        self.src_dir = "/"
        self.data_dir = os.path.join(self.src_dir, "data")
        self.plot_dir = os.path.join(self.src_dir, "plots")
        self.disk_dir = ""
        self.data_dir = ""

        # data
        self.agri_limitations_1M_path = os.path.join(self.data_dir, "land_lim", "cli_land_limitation_for_ag_1M.geojson")
        self.agri_limitations_250k_path = os.path.join(self.data_dir, "land_lim",
                                                       "cli_land_capability_for_ag_250K.geojson")
        self.biomass_inventory_mapping = os.path.join(self.data_dir, "biomass", "BIOMASS_INV_CT_GEOJSON.geojson")
        self.land_practices = os.path.join(self.data_dir, "land_practices", "AGR_MAJOR_LAND_PRACTICE_GROUPS.geojson")
        self.wildlife_habitat_connectivity = os.path.join(self.data_dir, "wildlife_habitat_connectivity",
                                                          "AEI_MGMNT_WHAF.geojson")
        self.ecological_connectivity_json = os.path.join(self.data_dir, "ecological_connectivity",
                                                         "eco_con.geojson")
        self.compiled_json = os.path.join(self.disk_dir, 'compiled.geojson')
        self.compiled_reduced_json = os.path.join(self.data_dir, 'compiled_reduced_cp.geojson')
        self.compiled_gpd_json = os.path.join(self.data_dir, 'compiled_gpd.geojson')
        self.compiled_gpd_use_json = os.path.join(self.data_dir, 'compiled_gpd_use_2.geojson')

        self.ag_ghg = os.path.join(self.data_dir, "ag_ghg", "AEI_AIR_GHG.geojson")
        self.ecological_connectivity_nc = os.path.join(self.data_dir, "ecological_connectivity",
                                                       "eco_con_r.nc")
        self.ecological_connectivity_tiff = os.path.join(self.data_dir, "ecological_connectivity",
                                                         "Current_Density_2_Resistance.tif")
        self.crop_yields = os.path.join(self.data_dir, "crop_yields")
        self.crop_inventory = os.path.join(self.data_dir, "crop_inventory")

        
        self.data_streams = ["agri_lims", "wf_hb_capacity", "biomass_inv", "land_practices"]
        self.all_props = ["eco_con", "MAJOR1", "MINOR1", "DESC_EN", "WHAF_2015_CLASS", "WHAF_2015_CLASS_EN",
                          "WHAF_2015_VAL", "WHAF_00_15_CHG_VAL", "WHAF_00_15_CHG_CLASS", "WHAF_00_15_CHG_CLASS_EN",
                          "CROP_BARLEY_YLD", "CROP_OAT_YLD", "CROP_WHEAT_YLD", "CROP_FLAX_YLD", "CROP_CORN_YLD",
                          "CROP_CANOLA_YLD", "CROP_SOYBEAN_YLD", "CROP_BARLEY_QTY", "CROP_OAT_QTY", "CROP_WHEAT_QTY",
                          "CROP_FLAX_QTY", "CROP_CORN_QTY", "CROP_CANOLA_QTY", "CROP_SOYBEAN_QTY",
                          "MAJOR_LAND_PRACTICES_GROUP_NUM"]
        self.data_keys = {"agri_lims": {"MAJOR1": [], "MINOR1": [], "DESC_EN": []},
                          "wf_hb_capacity": {"WHAF_2015_CLASS": [], "WHAF_2015_CLASS_EN": [], "WHAF_2015_VAL": [],
                                             "WHAF_00_15_CHG_VAL": [], "WHAF_00_15_CHG_CLASS": [],
                                             "WHAF_00_15_CHG_CLASS_EN": []},
                          "biomass_inv": {"CROP_BARLEY_YLD": [], "CROP_OAT_YLD": [], "CROP_WHEAT_YLD": [],
                                          "CROP_FLAX_YLD": [], "CROP_CORN_YLD": [], "CROP_CANOLA_YLD": [],
                                          "CROP_SOYBEAN_YLD": [], "CROP_BARLEY_QTY": [], "CROP_OAT_QTY": [],
                                          "CROP_WHEAT_QTY": [], "CROP_FLAX_QTY": [], "CROP_CORN_QTY": [],
                                          "CROP_CANOLA_QTY": [], "CROP_SOYBEAN_QTY": []},
                          "land_practices": {"MAJOR_LAND_PRACTICES_GROUP_NUM": []}}
        self.default_vals = {"agri_lims": {"MAJOR1": 0, "MINOR1": 0, "DESC_EN": ""},
                             "wf_hb_capacity": {"WHAF_2015_CLASS": 0, "WHAF_2015_CLASS_EN": "", "WHAF_2015_VAL": 0,
                                                "WHAF_00_15_CHG_VAL": 0, "WHAF_00_15_CHG_CLASS": 0,
                                                "WHAF_00_15_CHG_CLASS_EN": ""},
                             "biomass_inv": {"CROP_BARLEY_YLD": 0, "CROP_OAT_YLD": 0, "CROP_WHEAT_YLD": 0,
                                             "CROP_FLAX_YLD": 0, "CROP_CORN_YLD": 0, "CROP_CANOLA_YLD": 0,
                                             "CROP_SOYBEAN_YLD": 0, "CROP_BARLEY_QTY": 0, "CROP_OAT_QTY": 0,
                                             "CROP_WHEAT_QTY": 0, "CROP_FLAX_QTY": 0, "CROP_CORN_QTY": 0,
                                             "CROP_CANOLA_QTY": 0, "CROP_SOYBEAN_QTY": 0},
                             "land_practices": {"MAJOR_LAND_PRACTICES_GROUP_NUM": 0}}

        # Define parameters (crop-dependent)
        self.params = {
            'crops': {
                'Spring wheat': {
                    'margin': {
                        'alpha': 0.05,
                        'beta': 0.01,
                        'gamma': 0.2,
                        'delta': 0.05,
                        'epsilon': 0.01,
                        'zeta': 0.2
                    },
                    'habitat': {
                        'alpha': 0.05,
                        'beta': 0.005,
                        'gamma': 0.2,
                        'delta': 0.05,
                        'epsilon': 0.005,
                        'zeta': 0.2
                    },
                    'p_c': 200 #price in USD/Tonne
                },
                'Barley': {
                    'margin': {
                        'alpha': 0.05,
                        'beta': 0.01,
                        'gamma': 0.2,
                        'delta': 0.05,
                        'epsilon': 0.01,
                        'zeta': 0.2
                    },
                    'habitat': {
                        'alpha': 0.05,
                        'beta': 0.005,
                        'gamma': 0.2,
                        'delta': 0.05,
                        'epsilon': 0.005,
                        'zeta': 0.2
                    },
                    'p_c': 120 #price in USD/Tonne
                },
                'Canola/rapeseed': {
                    'margin': {
                        'alpha': 0.20,
                        'beta': 0.01,
                        'gamma': 0.2,
                        'delta': 0.10,
                        'epsilon': 0.01,
                        'zeta': 0.2
                    },
                    'habitat': {
                        'alpha': 0.20,
                        'beta': 0.005,
                        'gamma': 0.2,
                        'delta': 0.10,
                        'epsilon': 0.005,
                        'zeta': 0.2
                    },
                    'p_c': 1100 #price in USD/Tonne
                },
                'Corn': {
                    'margin': {
                        'alpha': 0.05,
                        'beta': 0.01,
                        'gamma': 0.2,
                        'delta': 0.05,
                        'epsilon': 0.01,
                        'zeta': 0.2
                    },
                    'habitat': {
                        'alpha': 0.05,
                        'beta': 0.005,
                        'gamma': 0.2,
                        'delta': 0.05,
                        'epsilon': 0.005,
                        'zeta': 0.2
                    },
                    'p_c': 190 #price in USD/Tonne
                },
                'Oats': {
                    'margin': {
                        'alpha': 0.05,
                        'beta': 0.01,
                        'gamma': 0.2,
                        'delta': 0.05,
                        'epsilon': 0.01,
                        'zeta': 0.2
                    },
                    'habitat': {
                        'alpha': 0.05,
                        'beta': 0.005,
                        'gamma': 0.2,
                        'delta': 0.05,
                        'epsilon': 0.005,
                        'zeta': 0.2
                    },
                    'p_c': 95 #price in USD/Tonne
                },
                'Soybeans': {
                    'margin': {
                        'alpha': 0.10,
                        'beta': 0.01,
                        'gamma': 0.2,
                        'delta': 0.10,
                        'epsilon': 0.01,
                        'zeta': 0.2
                    },
                    'habitat': {
                        'alpha': 0.10,
                        'beta': 0.005,
                        'gamma': 0.2,
                        'delta': 0.10,
                        'epsilon': 0.005,
                        'zeta': 0.2
                    },
                    'p_c': 370 #price in USD/Tonne
                }
            },
            'habitats': [
                "Broadleaf", "Coniferous", "Exposed land/barren",
                "Grassland", "Shrubland", "Water", "Wetland"
            ],
            'r': 0.05,  # 5% discount rate
            't': 20,  # 20-year time horizon
            'costs': {
                'margin': {
                    'implementation': 400,  # USD/ha one-time cost
                    'maintenance': 60  # USD/ha/year
                },
                'habitat': {
                    'implementation': 300,  # USD/ha one-time cost
                    'maintenance': 70,  # USD/ha/year
                    'existing_hab': 0 # USD/ha/year
                },
                'agriculture': {
                    'maintenance': 100  # USD/ha/year baseline maintenance cost
                }
            }
        }
