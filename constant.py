from os.path import join

ROOT_DATASET_DIR = r'E:\Unvertità\progettotesi\Dataset'
ant = join(ROOT_DATASET_DIR, r'ant')
ivy = join(ROOT_DATASET_DIR, 'ivy')
jedit = join(ROOT_DATASET_DIR, 'jedit')
lucene = join(ROOT_DATASET_DIR, 'lucene')
poi = join(ROOT_DATASET_DIR, 'poi')
synapse = join(ROOT_DATASET_DIR, 'synapse')
velocity = join(ROOT_DATASET_DIR, 'velocity')
xalan = join(ROOT_DATASET_DIR, 'xanal')
xcerces = join(ROOT_DATASET_DIR, 'xcerces')


needed = ['name', 'wmc', 'dit', 'noc', 'cbo', 'rfc', 'lcom',
       'ca', 'ce', 'npm', 'lcom3', 'loc', 'dam', 'moa', 'mfa', 'cam', 'ic',
       'cbm', 'amc', 'max_cc', 'avg_cc', 'bug']
features_withbug = ['wmc', 'dit', 'noc', 'cbo', 'rfc', 'lcom',
       'ca', 'ce', 'npm', 'lcom3', 'loc', 'dam', 'moa', 'mfa', 'cam', 'ic',
       'cbm', 'amc', 'max_cc', 'avg_cc', 'bug']
features = ['wmc', 'dit', 'noc', 'cbo', 'rfc', 'lcom',
       'ca', 'ce', 'npm', 'lcom3', 'loc', 'dam', 'moa', 'mfa', 'cam', 'ic',
       'cbm', 'amc', 'max_cc', 'avg_cc']