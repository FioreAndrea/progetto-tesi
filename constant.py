from os.path import join

ROOT_DATASET_DIR = r'C:\Users\alf9n\Documents\GitHub\progetto-tesi\Dataset'
ant = join(ROOT_DATASET_DIR, 'ant')
camel = join(ROOT_DATASET_DIR, 'camel')
ivy = join(ROOT_DATASET_DIR, 'ivy')
jedit = join(ROOT_DATASET_DIR, 'jedit')
log4j = join(ROOT_DATASET_DIR, 'log4j')
lucene = join(ROOT_DATASET_DIR, 'lucene')
poi = join(ROOT_DATASET_DIR, 'poi')
synapse = join(ROOT_DATASET_DIR, 'synapse')
velocity = join(ROOT_DATASET_DIR, 'velocity')
xalan = join(ROOT_DATASET_DIR, 'xalan')
xerces = join(ROOT_DATASET_DIR, 'xerces')
prop = join(ROOT_DATASET_DIR, 'prop')


needed = ['name', 'wmc', 'dit', 'noc', 'cbo', 'rfc', 'lcom',
       'ca', 'ce', 'npm', 'lcom3', 'loc', 'dam', 'moa', 'mfa', 'cam', 'ic',
       'cbm', 'amc', 'max_cc', 'avg_cc', 'bug']
features_withbug = ['wmc', 'dit', 'noc', 'cbo', 'rfc', 'lcom',
       'ca', 'ce', 'npm', 'lcom3', 'loc', 'dam', 'moa', 'mfa', 'cam', 'ic',
       'cbm', 'amc', 'max_cc', 'avg_cc', 'bug']
features = ['wmc', 'dit', 'noc', 'cbo', 'rfc', 'lcom',
       'ca', 'ce', 'npm', 'lcom3', 'loc', 'dam', 'moa', 'mfa', 'cam', 'ic',
       'cbm', 'amc', 'max_cc', 'avg_cc']