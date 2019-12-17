from setuptools import setup, find_packages

runtime = {
    'pandas==0.24.2',
    'scikit-learn==0.21.3',
    'gensim==3.8.0',
    'tensorflow==1.14.0',
    'Flask==1.1.1'
    ##
}

develop = {
    'flake8',
    'pytest',
}

# entry_points = {
#     'console_scripts': [
#         'predict-all-kcs=src.main:predict_all_kcs',
#         'predict-updated-kcs=src.main:predict_updated_kcs'
#     ]
# }

if __name__ == "__main__":
    setup(
        name="kcs_candidate_predictor",
        version="1.0.0",
        description="",
        keywords='',
        packages=find_packages(exclude=['tests', '*.tests', '*.tests.*']),
        install_requires=list(runtime),
        extras_require={
            'develop': list(runtime | develop)
        },
        # entry_points=entry_points,
    )
