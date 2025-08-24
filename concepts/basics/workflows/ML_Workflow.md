
- Git Repo and Initialization
    Define the repo for UAT , DEVELOPMENT and PROD for CI/CD later.

- Experimentation and Model Building
1. Data gathering and understanding. 
2. Data Preprocessing. 
    2.1 Remove Noise from the data.
    2.2 Stopwords , Transformations like Normalization  
    

    2.3 Structure dataset for training purpose. (Data versioning - For Data drift and concept Drift.) 

3. EDA - Analysis of independent classes , i.e . (Text, Class)

4. Correlations/ 

4/5. Generate Embedding if required. 

5. Define Model Architecture (NN,or XGClassifier or Other Classifiers)
6. Evaluation 
7. Registering of the Model in MLFlow .
8. Save the model.


Inference 

9. Load the final model .
8. Create Class based approach to load the model . 
9. Create inference/ predict function using  FASTAPI or Functions to call . -> save the 
10. log with framework like pheniox for QA and utilize the enriched dataset. 

