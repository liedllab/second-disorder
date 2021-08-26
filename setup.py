from setuptools import setup

setup(
    name="second-disorder",
    version="2.0.0a",
    description="Compute second and first order entropies for Grid Inhomogeneous Solvation Theory",
    author="Franz Waibl",
    author_email="franz.waibl@uibk.ac.at",
    include_package_data=False,
    zip_safe=False,
    install_requires=['numpy', 'pandas', 'numba', 'mdtraj', 'scipy', 'GridDataFormats', 'gisttools', 'pykdtree'],
    setup_requires=['pytest_runner'],
    tests_require=['pytest'],
    py_modules=[
        "second_disorder.base",
        "second_disorder.entropy",
        "second_disorder.density",
        "second_disorder.conditional_density",
        "second_disorder.density_shells",
        "second_disorder.command_line",
        "second_disorder.trans_entropy",
    ],
    entry_points={
        # "console_scripts": ["second_disorder=second_disorder.command_line:main"]
        "console_scripts": [
            "second_disorder_entropy=second_disorder.entropy:main",
            "second_disorder_density=second_disorder.density:main",
            "second_disorder_conditional_density=second_disorder.conditional_density:main",
            "second_disorder_density_shells=second_disorder.density_shells:main",
            "second_disorder_trans_entropy=second_disorder.trans_entropy:main",
        ]
    },
)
