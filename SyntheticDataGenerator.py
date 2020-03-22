#!/usr/bin/env python
# coding: utf-8

# In[1]:


from fhir_parser import FHIR, Patient, Observation
from fhir_parser.patient import Patient, Name, Telecom, Communications, Extension, Identifier
from fhir_parser.observation import Observation, ObservationComponent
import numpy as np
import pandas as pd
import pickle
import re
from datetime import date, time


# In[2]:


# from sklearn.datasets import make_spd_matrix


# In[3]:


# import seaborn as sns


# In[20]:


class SyntheticDataGenerator():
    
    
    def __init__(self):
        self.fhir = FHIR()
        self.patients = self.fhir.get_all_patients()
        self.num_examples = len(self.patients)
        self.num_features = 11
        self.patient_record_columns = ["UUID", "Name", "Gender", "Birthdate", "MaritalStatus", "MultipleBirth",
          "CoreRace", "CoreEthnicity", "Birthsex", "disabilityAdjLifeYears",
          "QualAdjLifeYears"]
    
    
    def getPatientsDF(self):

        patients = self.patients
        patient_data = np.zeros((self.num_examples, self.num_features), dtype=object)
        
        for i in range(self.num_examples):
            patient = patients[i]
            patient_data[i][0] = patient.uuid
            patient_data[i][1] = patient.full_name()
            patient_data[i][2] = patient.gender
            patient_data[i][3] = patient.birth_date
            patient_data[i][4] = patient.marital_status.marital_status
            patient_data[i][5] = patient.multiple_birth
            patient_data[i][6] = patient.get_extension("us-core-race")
            patient_data[i][7] = patient.get_extension("us-core-ethnicity")
            patient_data[i][8] = patient.get_extension("us-core-birthsex")
            patient_data[i][9] = patient.get_extension("disability-adjusted-life-years")
            patient_data[i][10] = patient.get_extension("quality-adjusted-life-years")

        return pd.DataFrame(patient_data, columns=self.patient_record_columns)
    
    
    
    def createObservations(self):
        
        # patient_observations = []
        # for uuid in patients_df["UUID"]:
        #     patient_observations.append(fhir.get_patient_observations(uuid))
        with open("patient_obs.pkl","rb") as f:
            patient_observations = pickle.load(f)
        
        components = []
        for obs in patient_observations:
            for ob in obs:
                for component in ob.components:
                    components.append(component.display)
        
        components_arr = np.unique(components)

        for i in range(len(components_arr)):
            components_arr[i] = components_arr[i].replace("\u200b", "")

        components_arr = np.unique(components_arr)
        
        components_dict = {}
        for i in range(components_arr.size):
            components_dict[components_arr[i]] = i
        
        patient_observations_arr = np.empty((self.num_examples,components_arr.size))
        patient_observations_arr[:] = np.nan

        number_obs_arr = np.zeros(patient_observations_arr.shape, dtype=int)

        for obs in patient_observations:
            for ob in obs:
                for component in ob.components:
                    components.append(component.display)
                    
        for i in range(len(patient_observations)):
            for j in range(len(patient_observations[i])):
                for k in range(len(patient_observations[i][j].components)):
                    ob=patient_observations[i][j].components[k]
                    index = (i,components_dict[ob.display.replace("\u200b", "")])
                    if np.isnan(patient_observations_arr[index]):
                        patient_observations_arr[index] = ob.value
                    else:
                        patient_observations_arr[index]+=ob.value
                    number_obs_arr[index]+=1
    
        patient_mean_obs = patient_observations_arr/number_obs_arr
        patient_observations_df = pd.DataFrame(patient_mean_obs, columns=components_arr)
        
        return patient_observations_df
    
    
    def generateDataset(self, patient_observations_df):
        
        patients_df = self.getPatientsDF()        
        collengths = []
        for column in patient_observations_df.columns:
            if len(patient_observations_df[column].dropna())<100:
                collengths.append(column)

        patient_observations_df.drop(collengths, axis=1, inplace=True)
        full_patient_df = pd.concat([patients_df,patient_observations_df],axis=1)
        full_patient_df["Age"] = (date.today() - full_patient_df["Birthdate"]).dt.days
        full_patient_df["Gender"] = (full_patient_df["Gender"] == "male")
        full_patient_df["MaritalStatus"] = (full_patient_df["MaritalStatus"]=="M")
        full_patient_df["Birthsex"] = (full_patient_df["Birthsex"]=="M")
        full_patient_df.rename({"Gender":"is_male", "MaritalStatus":"is_married","Birthsex":"birthsex_is_male"},axis=1, inplace=True)
        full_patient_df["is_male"]=full_patient_df["is_male"].astype(int)
        full_patient_df["is_married"]=full_patient_df["is_married"].astype(int)
        full_patient_df["MultipleBirth"]=full_patient_df["MultipleBirth"].astype(int)
        full_patient_df["birthsex_is_male"]=full_patient_df["birthsex_is_male"].astype(int)
        
        patient_dataset = full_patient_df.copy()
        cols_to_drop = ["Name","is_male","Birthdate","is_married",
                        "CoreRace", "UUID", "CoreEthnicity", "MultipleBirth", "Body mass index (BMI) [Percentile] Per age and gender",
                       "Weight-for-length Per age and sex","RBC Auto (Bld) [#/Vol]"]
        patient_dataset["disabilityAdjLifeYears"] = patient_dataset["disabilityAdjLifeYears"].astype(float)
        patient_dataset["QualAdjLifeYears"] = patient_dataset["QualAdjLifeYears"].astype(float)
        patient_dataset.drop(cols_to_drop,axis=1,inplace=True)
        return patient_dataset
      
    def generateSyntheticData(self, number_of_datapoints):
        
        patient_dataset = self.generateDataset(self.createObservations())
        
        covariance_data = patient_dataset.cov() + 45*np.identity(48)

        covariance_data.fillna(0, inplace=True)

        patient_cholesky = np.linalg.cholesky(covariance_data)

        samples = np.random.multivariate_normal(patient_dataset.mean().values, covariance_data, size=number_of_datapoints*2)
        synthetic_df = pd.DataFrame(samples)
        synthetic_df[0] = (synthetic_df[0]>=1)
        synthetic_df = synthetic_df.astype(int)
        synthetic_df = synthetic_df[(synthetic_df[1]>=0) & (synthetic_df[2]>=0)]
        synthetic_df[synthetic_df<0] = np.nan
        synthetic_df.columns=patient_dataset.columns
        synthetic_df["Age"] = synthetic_df["Age"]//365
        
        return synthetic_df.head(number_of_datapoints)


# In[ ]:




