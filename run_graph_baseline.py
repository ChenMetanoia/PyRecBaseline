import argparse
import logging
from recbole_gnn.config import Config
from recbole_gnn.utils import create_dataset, data_preparation, get_trainer
from recbole.utils import init_seed, ensure_dir
from utils.logger import init_logger
from utils.util import import_class, data_preparation, ColdEvaluator


def run(config_dict):
    config_dict[
        "checkpoint_dir"
    ] = f"results/{config_dict['PLM']}/{config_dict['dataset']}/graph/"

    # model initialization
    model_class = import_class(config_dict["model"])
    config_file_list = ["properties/train.yaml", "properties/data.yaml"]
    if config_dict["model"] == "LightGCN":
        config_file_list.append("properties/lightgcn.yaml")
    elif config_dict["model"] == "GAT":
        config_file_list.append("properties/gat.yaml")
    elif config_dict["model"] == "Transformer":
        config_file_list.append("properties/transformer.yaml")
    config = Config(
        model=model_class, config_dict=config_dict, config_file_list=config_file_list
    )
    # init random seed
    init_seed(config["seed"], config["reproducibility"])

    # logger initialization
    ensure_dir(config["checkpoint_dir"])
    init_logger(config)
    logger = logging.getLogger()

    # write config info into log
    logger.info(config)

    # dataset creating and filtering
    dataset = create_dataset(config)
    logger.info(dataset)

    # dataset splitting
    if "cold" in config["benchmark_filename"]:
        train_data, valid_data, test_data, cold_data = data_preparation(config, dataset)
    else:
        train_data, valid_data, test_data = data_preparation(config, dataset)

    model = model_class(config, train_data.dataset).to(config["device"])
    logger.info(model)

    # trainer loading and initialization
    trainer = get_trainer(config["MODEL_TYPE"], config["model"])(config, model)

    # model training
    if config["model"] != "LightGCN":
        best_valid_score, best_valid_result = trainer.fit(train_data, valid_data)
        load_best_model = True
    elif config["model"] == "LightGCN":
        if config["fix_item_emb"] and config["fix_user_emb"]:
            load_best_model = False
            pass
        else:
            load_best_model = True
            best_valid_score, best_valid_result = trainer.fit(train_data, valid_data)

    # model evaluation
    test_result = trainer.evaluate(test_data, load_best_model=load_best_model)
    res_str = str()
    for k, v in test_result.items():
        res_str += str(v) + " "
    logger.info("CSV_easy_copy_format:\n%s", res_str)

    if "cold" in config["benchmark_filename"]:
        cold_evaluator = ColdEvaluator(trainer, load_best_model=load_best_model)
        cold_test_result = cold_evaluator.evaluate_cold_item(
            cold_data, comparewith="item_emb", agg_neighbor=False
        )
        print("item cold start evaluation")
        res_str = str()
        for k, v in cold_test_result.items():
            res_str += str(v) + " "
        print("cold start, item embedding:", res_str)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpu_id", default="0", type=str)
    parser.add_argument(
        "--fix_item_emb", action="store_true", help="whether to fix item embedding"
    )
    parser.add_argument(
        "--fix_user_emb", action="store_true", help="whether to fix user embedding"
    )
    parser.add_argument("--num_layers", default=2, type=int, help="number of layers")
    parser.add_argument(
        "--in_channels", default=128, type=int, help="input embedding dimension"
    )
    parser.add_argument(
        "--hidden_channels", default=128, type=int, help="hidden embedding dimension"
    )
    parser.add_argument(
        "--out_channels", default=128, type=int, help="output embedding dimension"
    )
    parser.add_argument(
        "--adj_mat_type",
        default="norm",
        type=str,
        help="adjacency matrix type, [norm, rating_norm]",
    )
    parser.add_argument("--model", default="LightGCN", type=str, help="model name")
    parser.add_argument(
        "--PLM",
        default="instructor-xl",
        help="[instructor-xl, all-MiniLM-L6-v2, bert], None is baseline",
    )
    parser.add_argument(
        "--learning_rate", default=0.001, type=float, help="learning rate"
    )
    parser.add_argument("--loss_type", default="BPR", type=str, help="loss type")
    args = parser.parse_args()
    # configurations initialization
    config_dict = vars(args)
    config_dict["data_path"] = f"dataset/{args.PLM}"
    config_dict["benchmark_filename"] = ["train", "valid", "test", "cold"]
    if args.PLM == "instructor-xl" or args.PLM == "bert":
        item_embedding_dim = 768  # Instructor
    else:
        item_embedding_dim = 384  # sentenceBERT
    if args.PLM:
        config_dict["embedding_size"] = item_embedding_dim
        config_dict["in_channels"] = item_embedding_dim
        config_dict["hidden_channels"] = item_embedding_dim
        config_dict["out_channels"] = item_embedding_dim

    run(config_dict)
