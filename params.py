# params = {'BookOrder': {'penalty_time': 1,
#                         'penalty_mark': 1,
#                         'omega': 0.1,
#                         'v': 1},
#           'Club':
#               {'penalty_time': 5, 'penalty_mark': 0.1, 'omega': 5, 'v': 1},
#               # [
#               #     {
#               #         'penalty_time': 5,
#               #         'penalty_mark': 5,
#               #         'omega': 1,
#               #         'v': 1
#               #     },
#               #     {
#               #         'penalty_time': 1,
#               #         'penalty_mark': 1,
#               #         'omega': 0.1,
#               #         'v': 0.1
#               #     },
#               #     {
#               #         'penalty_time': 2,
#               #         'penalty_mark': 2,
#               #         'omega': 0.2,
#               #         'v': 0.2
#               #     },
#               #     {
#               #         'penalty_time': 0.5,
#               #         'penalty_mark': 0.5,
#               #         'omega': 1,
#               #         'v': 1
#               #     }],
#           # {'penalty_time': 1,
#           #      'penalty_mark': 1,
#           #      'omega': 0.1,
#           #      'v': 1},
#           'Election': {'penalty_time': 1,
#                        'penalty_mark': 1,
#                        'omega': 0.1,
#                        'v': 1},
#           'Series': {'penalty_time': 1,
#                      'penalty_mark': 1,
#                      'omega': 0.1,
#                      'v': 1},
#           'Verdict':
#               {
#                   'penalty_time': 2,
#                   'penalty_mark': 2,
#                   'omega': 0.2,
#                   'v': 0.2
#               },
#           }

# params = {
#     'Club': {'penalty_time': [0.1, 1], 'penalty_mark': [0.1, 1], 'omega': [0.1, 1], 'v': [0.1, 1]},
#     'Election': {'penalty_time': [0.1, 1], 'penalty_mark': [0.1, 1], 'omega': [0.1, 1], 'v': [0.1, 1]},
#     'Series': {'penalty_time': [0.1, 1], 'penalty_mark': [0.1, 1], 'omega': [0.1, 1], 'v': [0.1, 1]},
#     'Verdict': {'penalty_time': [0.1, 1], 'penalty_mark': [0.1, 1], 'omega': [0.1, 1], 'v': [0.1, 1]},
# }

params_classification = {
    # {'penalty_time': 1,
    #           'penalty_mark': 1,
    #           'omega': 0.1,
    #           'v': 1},
    # 'BookOrder': {'penalty_time': 0.1,
    #               'penalty_mark': 0.1,
    #               'omega': 0.2,
    #               'v': 0.2,
    #               'skip_first': False},

    'BookOrder': {'penalty_time': 5,
                  'penalty_mark': 5,
                  'omega': 5,
                  'v': 5,
                  'skip_first': True,
                  'stochastic_size': 20},
    'Club': {'penalty_time': 5,
             'penalty_mark': 5,
             'omega': 5,
             'v': 5,
             'skip_first': True,
             'stochastic_size': 20},
    'Election': {'penalty_time': 5,
                 'penalty_mark': 5,
                 'omega': 5,
                 'v': 5,
                 'skip_first': True,
                 'stochastic_size': 20},
    'Series': {'penalty_time': 2,
               'penalty_mark': 2,
               'omega': 0.1,
               'v': 0.1,
               'skip_first': False,
               'stochastic_size': 20},
    'Verdict':
        {
            'penalty_time': 5,
            'penalty_mark': 5,
            'omega': 0.5,
            'v': 0.5,
            'skip_first': False,
            'stochastic_size': 20
        },
    # 'Verdict':
    #     {
    #         'penalty_time': [0.1, 1, 5],
    #         'penalty_mark': [0.1, 1, 5],
    #         'omega': [0.05, 0.1],
    #         'v': [0.1, 0.5, 5],
    #         'skip_first': [True, False],
    #         'stochastic_size': 20
    #     },
    # 'Verdict':
    #     {
    #         'penalty_time': [1],
    #         'penalty_mark': [1],
    #         'omega': [5],
    #         'v': [5],
    #         'skip_first': [False],
    #         'stochastic_size': 20
    #     },
}
