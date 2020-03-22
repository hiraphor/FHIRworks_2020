# FHIR-Synthetic-Data-Generator
This project draws from advanced topics in Statistics and Linear Algebra in order to generate synthetic data from
a covariance matrix derived from patient data, and in doing so preserving sample characteristics and correlations.
To be exact, random multivariate normals are transformed to match the desired covariance matrix of the sample data.
A suitable transformation is found via a Cholesky decomposition of the original data, which is then multiplied against
random normals, resulting in the desired population characteristics.


## Prerequisites
```
Python 3.7
Numpy
Pandas
FHIR-Parser
```

## Using the generator
The generator relies on the FHIR-service, which should be running. The generator class can be imported and used as such:

```
from SyntheticDataGenerator import SyntheticDataGenerator as sdg
datagen = sdg()
patient_data = datagen.createObservations()
syntheticdata = datagen.generateSyntheticData(1000)
```

This will return syntheticdata, a Pandas DataFrame with 1,000 patient records. Data which has no correlations of any kind,
such as name and address, and some data which has strong multicollinearity, has been omitted. 



## Acknowledgements
Many thanks to the FHIR-Parser project, which enabled easier development.
https://pypi.org/project/FHIR-Parser/


## Authors
* **Louis Phillips**

##Licence
This project is licenced under the Apache Licence version 2.
