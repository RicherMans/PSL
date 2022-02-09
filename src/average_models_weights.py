import argparse
import torch


def process(models, output):
    model_res_state_dict = {}
    state_dict = {}
    for m in models:
        cur_state = torch.load(m, map_location='cpu')
        model_params = cur_state.pop('model')
        # Append non "model" items, encoder, optimizer etc ...
        for k in cur_state:
            state_dict[k] = cur_state[k]
        # Accumulate statistics
        for k in model_params:
            if k in model_res_state_dict:
                model_res_state_dict[k] += model_params[k]
            else:
                model_res_state_dict[k] = model_params[k]
    # Average
    for k in model_res_state_dict:
        # If there are any parameters
        if model_res_state_dict[k].ndim > 0:
            model_res_state_dict[k] /= float(len(models))
    state_dict['model'] = model_res_state_dict

    torch.save(state_dict, output)




if __name__ == "__main__":
    parser =argparse.ArgumentParser()
    parser.add_argument('models', nargs="+")
    parser.add_argument('-o',
                        '--output',
                        required=True,
                        help="Output model (pytorch)")
    args = parser.parse_args()
    process(**vars(args))
