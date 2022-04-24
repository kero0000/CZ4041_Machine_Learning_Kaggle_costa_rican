import pandas as pd
import datetime
import numpy as np

def load_processed_csv(train=True):
    '''train->label 1-4 becomes 0-3'''
    if train:
        data_df=pd.read_csv("data/train_processed.csv")
    else:
        data_df=pd.read_csv("data/test_processed.csv")
    # drop columns that are squared of the original features because those are correlated to the original anyway
    data_df = data_df[data_df.columns.drop(list(data_df.filter(regex='squared')))]

    # Some of the household features are the same or correlated, hence remove them
    data_df = data_df.drop(columns = ['Total persons in the household','size of the household','# of total individuals in the household'])

    # Urban and rural features are mutually exclusive so just need 1 of them
    data_df = data_df.drop(columns = ['=2 zona rural'])

    # need replace yes and no to be 1 and 0 respectively for some of the features
    mapping = {"yes": 1, "no": 0}

    data_df['Dependency rate'] = data_df['Dependency rate'].replace(mapping).astype(np.float64)
    data_df['years of education of female head of household'] = data_df['years of education of female head of household'].replace(mapping).astype(np.float64)
    data_df['years of education of male head of household'] = data_df['years of education of male head of household'].replace(mapping).astype(np.float64)

    # We can create ordinal variables for the walls, floor and roof conditions instead of 3 features each
    data_df['walls'] = np.argmax(np.array(data_df[['=1 if walls are bad','=1 if walls are regular','=1 if walls are good',]]),axis = 1)
    data_df['roof'] = np.argmax(np.array(data_df[[ '=1 if roof are bad','=1 if roof are regular','=1 if roof are good']]),axis = 1)
    data_df['floor'] = np.argmax(np.array(data_df[[ '=1 if floor are bad','=1 if floor are regular','=1 if floor are good']]),axis = 1)
    data_df = data_df.drop(columns = ['=1 if walls are bad','=1 if walls are regular','=1 if walls are good','=1 if roof are bad','=1 if roof are regular','=1 if roof are good', '=1 if floor are bad','=1 if floor are regular','=1 if floor are good'  ])

    # Similarly we can create ordinal variables for education level
    data_df['Education level'] = np.argmax(np.array(data_df[[ '=1 no level of education',
    '=1 incomplete primary',
    '=1 complete primary',
    '=1 incomplete academic secondary level',
    '=1 complete academic secondary level',
    '=1 incomplete technical secondary level',
    '=1 complete technical secondary level',
    '=1 undergraduate and higher education',
    '=1 postgraduate higher education']]),axis = 1)
    data_df = data_df.drop(columns = ['=1 no level of education',
    '=1 incomplete primary',
    '=1 complete primary',
    '=1 incomplete academic secondary level',
    '=1 complete academic secondary level',
    '=1 incomplete technical secondary level',
    '=1 complete technical secondary level',
    '=1 undergraduate and higher education',
    '=1 postgraduate higher education'])


    # house owndership can also be ordinal variables
    data_df['House Ownership'] = np.argmax(np.array(data_df[[ '=1 own and fully paid house',
    '=1 own, paying in installments',
    '=1 rented',
    '=1 precarious']]),axis = 1)
    data_df = data_df.drop(columns = [ '=1 own and fully paid house',
    '=1 own, paying in installments',
    '=1 rented',
    '=1 precarious'])

    #region ordinal variables
    data_df['Region '] = np.argmax(np.array(data_df[[  '=1 region Central',
    '=1 region Chorotega',
    '=1 region PacÃƒÂ\xadfico central',
    '=1 region Brunca',
    '=1 region Huetar AtlÃƒÂ¡ntica',
    '=1 region Huetar Norte']]),axis = 1)
    data_df = data_df.drop(columns = [  '=1 region Central',
    '=1 region Chorotega',
    '=1 region PacÃƒÂ\xadfico central',
    '=1 region Brunca',
    '=1 region Huetar AtlÃƒÂ¡ntica',
    '=1 region Huetar Norte'])

    data_df['Rubbish Disposal'] = np.argmax(np.array(data_df[[ '=1 if rubbish disposal mainly by tanker truck',
    '=1 if rubbish disposal mainly by botan hollow or buried',
    '=1 if rubbish disposal mainly by burning',
    '=1 if rubbish disposal mainly by throwing in an unoccupied space',
    '=1 if rubbish disposal mainly by throwing in river, creek or sea',
    '=1 if rubbish disposal mainly other']]),axis = 1)
    data_df = data_df.drop(columns = [ '=1 if rubbish disposal mainly by tanker truck',
    '=1 if rubbish disposal mainly by botan hollow or buried',
    '=1 if rubbish disposal mainly by burning',
    '=1 if rubbish disposal mainly by throwing in an unoccupied space',
    '=1 if rubbish disposal mainly by throwing in river, creek or sea',
    '=1 if rubbish disposal mainly other'])

    data_df['Energy Usage'] = np.argmax(np.array(data_df[['=1 no main source of energy used for cooking (no kitchen)',
    '=1 main source of energy used for cooking electricity',
    '=1 main source of energy used for cooking gas',
    '=1 main source of energy used for cooking wood charcoal']]),axis = 1)
    data_df = data_df.drop(columns = ['=1 no main source of energy used for cooking (no kitchen)',
    '=1 main source of energy used for cooking electricity',
    '=1 main source of energy used for cooking gas',
    '=1 main source of energy used for cooking wood charcoal'])

    data_df['Toilet Pipeline'] = np.argmax(np.array(data_df[[ '=1 no toilet in the dwelling',
    '=1 toilet connected to sewer or cesspool',
    '=1 toilet connected to  septic tank',
    '=1 toilet connected to black hole or letrine',
    '=1 toilet connected to other system']]),axis = 1)
    data_df = data_df.drop(columns = [ '=1 no toilet in the dwelling',
    '=1 toilet connected to sewer or cesspool',
    '=1 toilet connected to  septic tank',
    '=1 toilet connected to black hole or letrine',
    '=1 toilet connected to other system'])

    data_df['Outside Wall Material'] = np.argmax(np.array(data_df[[ '=1 if predominant material on the outside wall is block or brick',
    '=1 if predominant material on the outside wall is socket (wood, zinc or absbesto',
    '=1 if predominant material on the outside wall is prefabricated or cement',
    '=1 if predominant material on the outside wall is waste material',
    '=1 if predominant material on the outside wall is wood ',
    '=1 if predominant material on the outside wall is zink',
    '=1 if predominant material on the outside wall is natural fibers',
    '=1 if predominant material on the outside wall is other']]),axis = 1)
    data_df = data_df.drop(columns = [ '=1 if predominant material on the outside wall is block or brick',
    '=1 if predominant material on the outside wall is socket (wood, zinc or absbesto',
    '=1 if predominant material on the outside wall is prefabricated or cement',
    '=1 if predominant material on the outside wall is waste material',
    '=1 if predominant material on the outside wall is wood ',
    '=1 if predominant material on the outside wall is zink',
    '=1 if predominant material on the outside wall is natural fibers',
    '=1 if predominant material on the outside wall is other'])

    data_df['Floor Material'] = np.argmax(np.array(data_df[[ '=1 if predominant material on the floor is mosaic, ceramic, terrazo',
    '=1 if predominant material on the floor is cement',
    '=1 if predominant material on the floor is other',
    '=1 if predominant material on the floor is  natural material',
    '=1 if no floor at the household',
    '=1 if predominant material on the floor is wood']]),axis = 1)
    data_df = data_df.drop(columns = [ '=1 if predominant material on the floor is mosaic, ceramic, terrazo',
    '=1 if predominant material on the floor is cement',
    '=1 if predominant material on the floor is other',
    '=1 if predominant material on the floor is  natural material',
    '=1 if no floor at the household',
    '=1 if predominant material on the floor is wood'])

    data_df['Roof Material'] = np.argmax(np.array(data_df[[ '=1 if predominant material on the roof is metal foil or zink',
    '=1 if predominant material on the roof is fiber cement, mezzanine ',
    '=1 if predominant material on the roof is natural fibers',
    '=1 if predominant material on the roof is other',
    '=1 if the house has ceiling']]),axis = 1)
    data_df = data_df.drop(columns = [ '=1 if predominant material on the roof is metal foil or zink',
    '=1 if predominant material on the roof is fiber cement, mezzanine ',
    '=1 if predominant material on the roof is natural fibers',
    '=1 if predominant material on the roof is other',
    '=1 if the house has ceiling'])

    data_df['Water Provision'] = np.argmax(np.array(data_df[[ '=1 if water provision inside the dwelling',
    '=1 if water provision outside the dwelling',
    '=1 if no water provision']]),axis = 1)
    data_df = data_df.drop(columns = [ '=1 if water provision inside the dwelling',
    '=1 if water provision outside the dwelling',
    '=1 if no water provision'])

    data_df['Electricity Provision'] = np.argmax(np.array(data_df[[ '=1 electricity from CNFL, ICE, ESPH/JASEC',
    '=1 electricity from private plant',
    '=1 no electricity in the dwelling',
    '=1 electricity from cooperative']]),axis = 1)
    data_df = data_df.drop(columns = [ '=1 electricity from CNFL, ICE, ESPH/JASEC',
    '=1 electricity from private plant',
    '=1 no electricity in the dwelling',
    '=1 electricity from cooperative','Id','Household level identifier'])# add 'Id','Household level identifier'
    if train:
        label=data_df['Target']
        data_df = data_df.drop('Target',axis=1)
        data_df.insert(len(data_df.columns),'label',label)
        data_df.columns=[f"feature{i}"for i in range(len(data_df.columns[:-1]))]+['label']#renmae the feature
        data_df['label']-=1# make label from 0-3 instead of 1-4
    return data_df



def generate_sumbit_csv(model,text=""):
    '''auto tranform label 0-3 to 1-4'''
    test_processed_df=load_processed_csv(train=False)
    X = test_processed_df
    if (text == "lightgbm") or (text == "XGBoost"):
        y_test_pred=model.predict(np.array(X)).argmax(axis=1)+1
    else:
        y_test_pred = model.predict(np.array(X))+1
    test_df=pd.read_csv('data/test.csv')
    test_df.insert(len(test_df.columns),'Target',y_test_pred)
    submit_df=pd.concat([test_df['Id'],test_df['Target']],axis=1)
    submit_df.to_csv(f'data/test_submit_{text}_{str(datetime.datetime.now())}.csv')
    print('save to '+f'data/test_submit_{text}_{str(datetime.datetime.now())}.csv')
    print('distribution:')
    for i in range(1,5):
        print(i,len(y_test_pred[y_test_pred==i]))
    return True
