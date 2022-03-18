from distutils.core import setup

setup(
    name='emote',
    version='0.1',
    description='A modular reinforcement learning library',
    author ='Martin Singh-Blom, Tom Solberg, Jack Harmer, Jorge Del Val, Riley Miladi',
    author_email='martin.singh-blom@embark-studios.com, tom.solberg@embark-studios.com, jack.harmer@embark-studios.com, jorge.delval@embark-studios.com, riley.miladi@embark-studios.com',
    packages=[],
    install_requires=[
        'gym',
        'gym[atari]',
        'gym[box2d]',
        'gym[classic_control]',
        'sphinx-rtd-theme',
        'black'
    ]
)