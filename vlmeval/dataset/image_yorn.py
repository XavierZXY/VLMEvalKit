from ..smp import *
from ..utils import *
from .image_base import ImageBaseDataset
from .utils import DEBUG_MESSAGE, build_judge


class ImageYORNDataset(ImageBaseDataset):
    TYPE = "Y/N"

    DATASET_URL = {
        "MME": "https://opencompass.openxlab.space/utils/VLMEval/MME.tsv",
        "HallusionBench": "https://opencompass.openxlab.space/utils/VLMEval/HallusionBench.tsv",
        "POPE": "https://opencompass.openxlab.space/utils/VLMEval/POPE.tsv",
        "AMBER": "https://huggingface.co/datasets/yifanzhang114/LMM_Eval/resolve/main/AMBER.tsv",
    }

    DATASET_MD5 = {
        # 'MME': 'b36b43c3f09801f5d368627fb92187c3',
        "MME": "feb2029633058c5539514267edf0f55e",
        "HallusionBench": "0c23ac0dc9ef46832d7a24504f2a0c7c",
        "POPE": "c12f5acb142f2ef1f85a26ba2fbe41d5",
        "AMBER": "f613a9111b1bdc6aa621c800d9c96a31",
    }

    # It returns a dataframe
    def evaluate(self, eval_file, **judge_kwargs):
        from .utils.yorn import (
            AMBER_rating,
            Hallusion_rating,
            MME_rating,
            POPE_rating,
            YOrN_auxeval,
            YOrN_Extraction,
            default_rating,
        )

        dataset = self.dataset_name
        data = load(eval_file)
        data["prediction"] = [str(x) for x in data["prediction"]]
        storage = eval_file.replace(".xlsx", "_auxmatch.xlsx")
        tmp_file = eval_file.replace(".xlsx", "_tmp.pkl")
        nproc = judge_kwargs.pop("nproc", 4)

        if not osp.exists(storage):
            ans_map = {
                k: YOrN_Extraction(v) for k, v in zip(data["index"], data["prediction"])
            }
            if osp.exists(tmp_file):
                tmp = load(tmp_file)
                for k in tmp:
                    if ans_map[k] == "Unknown" and tmp[k] != "Unknown":
                        ans_map[k] = tmp[k]

            data["extracted"] = [ans_map[x] for x in data["index"]]
            unknown = data[data["extracted"] == "Unknown"]

            model = judge_kwargs.get("model", "exact_matching")
            if model == "exact_matching":
                model = None
            elif gpt_key_set():
                model = build_judge(**judge_kwargs)
                if not model.working():
                    warnings.warn(
                        "OPENAI API is not working properly, will use exact matching for evaluation"
                    )
                    warnings.warn(DEBUG_MESSAGE)
                    model = None
            else:
                model = None
                warnings.warn(
                    "OPENAI_API_KEY is not working properly, will use exact matching for evaluation"
                )

            if model is not None:
                lt = len(unknown)
                lines = [unknown.iloc[i] for i in range(lt)]
                tups = [(model, line) for line in lines]
                indices = list(unknown["index"])
                if len(tups):
                    res = track_progress_rich(
                        YOrN_auxeval,
                        tups,
                        nproc=nproc,
                        chunksize=nproc,
                        keys=indices,
                        save=tmp_file,
                    )
                    for k, v in zip(indices, res):
                        ans_map[k] = v

            data["extracted"] = [ans_map[x] for x in data["index"]]
            dump(data, storage)

        data = load(storage)
        if listinstr(["AMBER"], dataset):
            data["score"] = data["answer"].str.lower() == data["extracted"].str.lower()
        else:
            data["score"] = data["answer"] == data["extracted"]
        dump(data, storage)

        if dataset is not None and listinstr(["MME"], dataset):
            score = MME_rating(storage)
        elif dataset is not None and listinstr(["Hallusion"], dataset):
            score = Hallusion_rating(storage)
        elif dataset is not None and listinstr(["POPE"], dataset):
            score = POPE_rating(storage)
        elif dataset is not None and listinstr(["AMBER"], dataset):
            score = AMBER_rating(storage)
        else:
            score = default_rating(storage)

        score_tgt = eval_file.replace(".xlsx", "_score.csv")
        dump(score, score_tgt)
        return score
