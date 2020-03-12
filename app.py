import pandas as pd
from flask import Flask, render_template,url_for,request
import pickle
import numpy as np
from sklearn.externals import joblib

app = Flask(__name__)


def preprocess(df):
    df1 = df.drop(['weight','payer_code','medical_specialty'], axis = 1)
    df1 = df1.drop(['examide','citoglipton'], axis=1)
    df.admission_type_id.replace(
    list(range(1,9)),['Emergency',
    'Urgent',
    'Elective',
    'Newborn',
    'Not Available',
    'NULL',
    'Trauma Center',
    'Not Mapped'], inplace=True)
    id_list = ['Physician Referral',
    'Clinic Referral',
    'HMO Referral',
    'Transfer from a hospital',
    'Transfer from a Skilled Nursing Facility (SNF)',
    'Transfer from another health care facility',
    'Emergency Room',
    'Court/Law Enforcement',
    'Not Available',
    'Transfer from critial access hospital',
    'Normal Delivery',
    'Premature Delivery',
    'Sick Baby',
    'Extramural Birth',
    'Not Available',
    'NULL',
    'Transfer From Another Home Health Agency',
    'Readmission to Same Home Health Agency',
    'Not Mapped',
    'Unknown/Invalid',
    'Transfer from hospital inpt/same fac reslt in a sep claim',
    'Born inside this hospital',
    'Born outside this hospital',
    'Transfer from Ambulatory Surgery Center',
    'Transfer from Hospice']

    df1.admission_source_id.replace(list(range(1,len(id_list)+1)),id_list, inplace=True)
    id_list = ['Discharged to home','Discharged/transferred to another short term hospital','Discharged/transferred to SNF',
    'Discharged/transferred to ICF','Discharged/transferred to another type of inpatient care institution',
    'Discharged/transferred to home with home health service','Left AMA','Discharged/transferred to home under care of Home IV provider',
    'Admitted as an inpatient to this hospital',
    'Neonate discharged to another hospital for neonatal aftercare',
    'Expired',
    'Still patient or expected to return for outpatient services',
    'Hospice / home',
    'Hospice / medical facility',
    'Discharged/transferred within this institution to Medicare approved swing bed',
    'Discharged/transferred/referred another institution for outpatient services',
    'Discharged/transferred/referred to this institution for outpatient services',
    'NULL',
    'Expired at home. Medicaid only, hospice.',
    'Expired in a medical facility. Medicaid only, hospice.',
    'Expired, place unknown. Medicaid only, hospice.',
    'Discharged/transferred to another rehab fac including rehab units of a hospital .',
    'Discharged/transferred to a long term care hospital.',
    'Discharged/transferred to a nursing facility certified under Medicaid but not certified under Medicare.',
    'Not Mapped',
    'Unknown/Invalid',
    'Discharged/transferred to another Type of Health Care Institution not Defined Elsewhere',
    'Discharged/transferred to a federal health care facility.',
    'Discharged/transferred/referred to a psychiatric hospital of psychiatric distinct part unit of a hospital',
    'Discharged/transferred to a Critical Access Hospital (CAH).']

    df1.discharge_disposition_id.replace(list(range(1,len(id_list)+1)),id_list, inplace=True)
    df1 = df1[df1.discharge_disposition_id.str.contains("Expired") == False]
    df1 = df1[df1.discharge_disposition_id.str.contains("Hospice") == False]
    numeric_code_ranges = [(1,139),(140,239),(240,279),(280,289),(290,319),(320,389),(390,459),(460,519),(520,579),(580,629),
    (630,677),(680,709),(710,739),(740,759),  (760,779),  (780,799),(800,999)]
    ICD9_diagnosis_groups = ['Infectious And Parasitic Diseases',
    'Neoplasms',
    'Endocrine, Nutritional And Metabolic Diseases, And Immunity Disorders',
    'Diseases Of The Blood And Blood-Forming Organs',
    'Mental Disorders',
    'Diseases Of The Nervous System And Sense Organs',
    'Diseases Of The Circulatory System',
    'Diseases Of The Respiratory System',
    'Diseases Of The Digestive System',
    'Diseases Of The Genitourinary System',
    'Complications Of Pregnancy, Childbirth, And The Puerperium',
    'Diseases Of The Skin And Subcutaneous Tissue',
    'Diseases Of The Musculoskeletal System And Connective Tissue',
    'Congential Anomalies',
    'Certain Conditions Originating In The Perinatal Period',
    'Symptoms, Signs, And Ill-Defined Conditions',
    'Injury And Poisoning']
    codes = zip(numeric_code_ranges, ICD9_diagnosis_groups)
    codeSet = set(codes)
    df_icd = df1.copy()
    for num_range, diagnosis in codeSet:
        #print(num_range)
        oldlist = range(num_range[0],num_range[1]+1)
        oldlist = [x for x in oldlist]
        newlist = [diagnosis] * len(oldlist)
        for curr_col in ['diag_1', 'diag_2', 'diag_3']:
            df_icd[curr_col].replace(oldlist, newlist, inplace=True)
#             print(df_icd[curr_col])
    for curr_col in ['diag_1', 'diag_2', 'diag_3']:
        df_icd[curr_col].replace(oldlist, newlist, inplace=True)
        if('V' in str(df_icd[curr_col][0])):
            df_icd[curr_col] = 'Supplementary Classification Of Factors Influencing Health Status And Contact With Health Services'
#         df_icd.loc[df_icd[curr_col].str.contains("V"), curr_col] = 'Supplementary Classification Of Factors Influencing Health Status And Contact With Health Services'
    for curr_col in ['diag_1', 'diag_2', 'diag_3']:
        df_icd[curr_col].replace(oldlist, newlist, inplace=True)
        if('E' in str(df_icd[curr_col][0])):
             df_icd[curr_col] = 'Supplementary Classification Of External Causes Of Injury And Poisoning'
#         df_icd.loc[df_icd[curr_col].str.contains('E'), curr_col] = 'Supplementary Classification Of External Causes Of Injury And Poisoning'
    for curr_col in ['diag_1', 'diag_2', 'diag_3']:
        df_icd[curr_col].replace(oldlist, newlist, inplace=True)
        if('250' in str(df_icd[curr_col][0])):
             df_icd[curr_col] = 'Diabetes mellitus'
#         df_icd.loc[df_icd[curr_col].str.contains('250'), curr_col] ='Diabetes mellitus'
    
    df_icd = df_icd.replace('?',np.nan)
    cols_num = ['time_in_hospital','num_lab_procedures', 'num_procedures', 'num_medications',
       'number_outpatient', 'number_emergency', 'number_inpatient','number_diagnoses']
    df_icd['race'] = df_icd['race'].fillna('RACE')
    cols_cat = ['race', 'gender', 
       'max_glu_serum', 'A1Cresult',
       'metformin', 'repaglinide', 'nateglinide', 'chlorpropamide',
       'glimepiride', 'acetohexamide', 'glipizide', 'glyburide', 'tolbutamide',
       'pioglitazone', 'rosiglitazone', 'acarbose', 'miglitol', 'troglitazone',
       'tolazamide', 'insulin',
       'glyburide-metformin', 'glipizide-metformin',
       'glimepiride-pioglitazone', 'metformin-rosiglitazone',
       'metformin-pioglitazone', 'change', 'diabetesMed','admission_type_id',
        'discharge_disposition_id', 'admission_source_id']
    col1 = ['race_Asian', 'race_Caucasian', 'race_Hispanic', 'race_Other', 'race_RACE', 'gender_Male', 'gender_Unknown/Invalid',
            'max_glu_serum_>300', 'max_glu_serum_None', 'max_glu_serum_Norm', 'A1Cresult_>8', 'A1Cresult_None', 'A1Cresult_Norm',
            'metformin_No', 'metformin_Steady', 'metformin_Up', 'repaglinide_No', 'repaglinide_Steady', 'repaglinide_Up', 
            'nateglinide_No', 'nateglinide_Steady', 'nateglinide_Up', 'chlorpropamide_No', 'chlorpropamide_Steady', 
            'chlorpropamide_Up', 'glimepiride_No', 'glimepiride_Steady', 'glimepiride_Up', 'acetohexamide_Steady',
            'glipizide_No', 'glipizide_Steady', 'glipizide_Up', 'glyburide_No', 'glyburide_Steady', 'glyburide_Up',
            'tolbutamide_Steady', 'pioglitazone_No', 'pioglitazone_Steady', 'pioglitazone_Up', 'rosiglitazone_No',
            'rosiglitazone_Steady', 'rosiglitazone_Up', 'acarbose_No', 'acarbose_Steady', 'acarbose_Up', 'miglitol_No', 
            'miglitol_Steady', 'miglitol_Up', 'troglitazone_Steady', 'tolazamide_Steady', 'tolazamide_Up', 'insulin_No',
            'insulin_Steady', 'insulin_Up', 'glyburide-metformin_No', 'glyburide-metformin_Steady', 'glyburide-metformin_Up', 
            'glipizide-metformin_Steady', 'glimepiride-pioglitazone_Steady', 'metformin-rosiglitazone_Steady', 
            'metformin-pioglitazone_Steady', 'change_No', 'diabetesMed_Yes', 'admission_type_id_Emergency', 
            'admission_type_id_NULL', 'admission_type_id_Newborn', 'admission_type_id_Not Available', 
            'admission_type_id_Not Mapped', 'admission_type_id_Trauma Center', 'admission_type_id_Urgent',
            'discharge_disposition_id_Discharged to home', 'discharge_disposition_id_Discharged/transferred to ICF',
            'discharge_disposition_id_Discharged/transferred to SNF', 
            'discharge_disposition_id_Discharged/transferred to a federal health care facility.',
            'discharge_disposition_id_Discharged/transferred to a long term care hospital.', 
            'discharge_disposition_id_Discharged/transferred to a nursing facility certified under Medicaid but not certified under Medicare.', 'discharge_disposition_id_Discharged/transferred to another Type of Health Care Institution not Defined Elsewhere', 
            'discharge_disposition_id_Discharged/transferred to another rehab fac including rehab units of a hospital .', 
            'discharge_disposition_id_Discharged/transferred to another short term hospital',
            'discharge_disposition_id_Discharged/transferred to another type of inpatient care institution',
            'discharge_disposition_id_Discharged/transferred to home under care of Home IV provider', 
            'discharge_disposition_id_Discharged/transferred to home with home health service',
            'discharge_disposition_id_Discharged/transferred within this institution to Medicare approved swing bed', 
            'discharge_disposition_id_Discharged/transferred/referred another institution for outpatient services', 
            'discharge_disposition_id_Discharged/transferred/referred to this institution for outpatient services', 
            'discharge_disposition_id_Left AMA', 'discharge_disposition_id_NULL', 
            'discharge_disposition_id_Neonate discharged to another hospital for neonatal aftercare', 
            'discharge_disposition_id_Not Mapped', 'discharge_disposition_id_Still patient or expected to return for outpatient services',
            'admission_source_id_Clinic Referral', 'admission_source_id_Court/Law Enforcement', 'admission_source_id_Emergency Room', 
            'admission_source_id_Extramural Birth', 'admission_source_id_HMO Referral', 'admission_source_id_Normal Delivery',
            'admission_source_id_Not Available', 'admission_source_id_Physician Referral', 'admission_source_id_Sick Baby',
            'admission_source_id_Transfer From Another Home Health Agency', 'admission_source_id_Transfer from Hospice',
            'admission_source_id_Transfer from a Skilled Nursing Facility (SNF)', 'admission_source_id_Transfer from a hospital', 
            'admission_source_id_Transfer from another health care facility', 'admission_source_id_Transfer from critial access hospital',
            'admission_source_id_Unknown/Invalid']
    df_cat = df_icd[cols_cat]
    df_cat = pd.get_dummies(df_cat)
    df_cat = df_cat.reindex(columns = col1,fill_value=0)
    #print(df_cat)
    df_icd = df_icd.drop(cols_cat,axis=1)
    df_icd = pd.concat([df_icd,df_cat], axis = 1)
    cols_all_cat = list(df_cat.columns)
    
    age_id = {'[0-10)':0, 
          '[10-20)':10, 
          '[20-30)':20, 
          '[30-40)':30, 
          '[40-50)':40, 
          '[50-60)':50,
          '[60-70)':60, 
          '[70-80)':70, 
          '[80-90)':80, 
          '[90-100)':90}
    df_icd['age_group'] = df_icd.age.replace(age_id)
    cols_extra = ['age_group']
    col2 = cols_num + cols_all_cat + cols_extra
    #print(df_icd)
    df_data = df_icd[col2]
    return df_data

def predict_value():
    df_s = pd.read_csv('result.csv',index_col=False)
    dfk=preprocess(df_s)
    svm = joblib.load('model_svm.pkl')
    tree = joblib.load('model_tree.pkl')
    rf = joblib.load('model_rf.pkl')
    lr = joblib.load('model_lr.pkl')
    pred1 = svm.predict(dfk)
    pred2 = tree.predict(dfk)
    pred3 = rf.predict(dfk)
    pred4 = lr.predict(dfk)
    return pred1[0]+pred2[0]+pred3[0]+pred4[0]

@app.route('/')
@app.route('/predict')
def predict():
    result = predict_value()
    return render_template('predict.html',prediction = result)



if __name__ == '__main__':
     app.run(debug=True)


    
    