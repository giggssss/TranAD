import sys
import os
import asyncio

from dotenv import load_dotenv


load_dotenv(dotenv_path="./env/goyang_prod.env")

__dir = os.getenv("__DIR")

if __dir is not None:
    if __dir != os.getcwd():
        os.chdir(__dir)
else:
    os.environ["__DIR"] = os.getcwd()

DEV_ES_NODE = os.getenv("DEV_ES_NODE")
ES_NODE = os.getenv("ES_NODE")
WORKFLOW_TEMPLATEID = os.getenv("WORKFLOW_TEMPLATEID")
SITE_LABEL = os.getenv("SITE_LABEL")
PYTHON_ENV = os.getenv("PYTHON_ENV")


sys.path.append(os.getenv("__DIR"))

from modules.prophet.prophet_goyang import ProphetGoyang  # pylint: disable=E0401
from modules.db.es_client import ElasticClient  # pylint: disable=E0401


db_client = ElasticClient(node=ES_NODE)
dev_db_client = ElasticClient(node=DEV_ES_NODE) if (DEV_ES_NODE is not None) else None

aliot_prophet = ProphetGoyang(
    db_client=db_client, dev_db_client=dev_db_client, db_info_index=SITE_LABEL, db_data_index=f"{SITE_LABEL}-data"
)


async def create_ommited_device_model(device_ids: list, site_labels: list, dir_path: str):
    """
    데이터가 중간에 누락된 장치들의 모델을 생성
    """
    df_devices = await aliot_prophet.get_devices_from_db()
    print(df_devices)
    ommited_device_ids = device_ids
    ommited_devices = df_devices[df_devices["_id"].isin(ommited_device_ids)]
    origin_df = aliot_prophet.read_datas()

    df = None
    for site_label in site_labels:
        added_df = aliot_prophet.read_datas(site_label=site_label)
        df = origin_df.append(added_df, ignore_index=True)

    await aliot_prophet.create_model(devices=ommited_devices, train_data=df, save_path=dir_path)


async def run_ommited_device_model(
    template_id: str, round: int, dir_path: str = os.path.join(os.getenv("__DIR"), "models", "seperate"), except_device_ids: list = None
):
    """
    데이터가 중간에 누락된 장치 모델을 실행
    """
    if except_device_ids is None:
        except_device_ids = []

    models_by_name = aliot_prophet.read_models(dir_path=dir_path)
    print(len(models_by_name))

    def filter_devices(device_ids: list, model_name):
        is_filter_device = any(device_id in model_name for device_id in device_ids)
        if is_filter_device:
            return False
        return True

    models_by_name = dict(filter(lambda model: filter_devices(model_name=model[0], device_ids=except_device_ids), models_by_name.items()))

    print(len(models_by_name))
    print(models_by_name.keys())
    await aliot_prophet.run_model(template_id=template_id, round=round, models_by_name=models_by_name)


async def update_ommited_device_round(cur_round: int, update_round: int, device_ids: list):
    await aliot_prophet.update_round(device_ids=device_ids, cur_round=cur_round, update_round=update_round)


async def main():
    aliot_prophet.set_logger(dir_path="./logs")

    ommited_device_ids = [
        "UPOD130BpjBV2zbVcKxp",
        "-PPE5X0BpjBV2zbVwqxy",
        "uPO_5X0BpjBV2zbV4Kxk",
        "-fN8130BpjBV2zbVzasW",
        "7vN7130BpjBV2zbV9qs6",
        "s_N3130BpjBV2zbVdau9",
        "APN9130BpjBV2zbVVazQ",
        "VvOD130BpjBV2zbV5ayU",
        "_vO9ZX4BpjBV2zbVwq0q",
        "UfOD130BpjBV2zbVg6zu",
        "WfOE130BpjBV2zbVIKwr",
        "bPOF130BpjBV2zbVk6xM",
        "yPPB5X0BpjBV2zbVGKzo",
        "NfPJ5X0BpjBV2zbVaq0F",
        "6fPD5X0BpjBV2zbVnax2",
        "n_OJ130BpjBV2zbVd6x9",
        "IvPH5X0BpjBV2zbV9q3e",
        "y_N5130BpjBV2zbVSquL",
    ]

    body = {
        "query": {
            "bool": {
                "must": [{"match": {"con.category.keyword": "forecast"}}],
                "must_not": [{"terms": {"con.deviceId.keyword": ommited_device_ids}}],
            }
        },
        "sort": [{"con.nodes.forecastDate": {"order": "desc"}}],
        "size": 1,
    }
    # Get Round
    last_round = await aliot_prophet.get_last_round(body=body)

    # create model
    # await aliot_prophet.create_model()
    # await create_ommited_device_model(
    #     device_ids=[
    #   "UPOD130BpjBV2zbVcKxp",
    #   "-PPE5X0BpjBV2zbVwqxy",
    #   "uPO_5X0BpjBV2zbV4Kxk",
    #   "-fN8130BpjBV2zbVzasW",
    #   "7vN7130BpjBV2zbV9qs6",
    #   "APN9130BpjBV2zbVVazQ",
    #   "VvOD130BpjBV2zbV5ayU",
    #   "_vO9ZX4BpjBV2zbVwq0q",
    #   "bPOF130BpjBV2zbVk6xM",
    #   "yPPB5X0BpjBV2zbVGKzo",
    #   "NfPJ5X0BpjBV2zbVaq0F",
    #   "6fPD5X0BpjBV2zbVnax2",
    # ],
    #     site_labels=["seperate"],
    #     dir_path=os.path.join(os.getenv("__DIR"), "models", "seperate"),
    # )
    # await create_ommited_device_model(
    #     device_ids=[
    #         "n_OJ130BpjBV2zbVd6x9",
    #         "UfOD130BpjBV2zbVg6zu",
    #         "PfPK5X0BpjBV2zbVBq1I",
    #         "__O9ZX4BpjBV2zbV1a2z",
    #     ],
    #     site_labels=["seperate", "etc"],
    #     dir_path=os.path.join(os.getenv("__DIR"), "models", "etc"),
    # )

    # run model
    # 10/3~ 분석시작 모델
    # await aliot_prophet.run_model(template_id=WORKFLOW_TEMPLATEID, round=last_round + 1)

    # 10/10~ 분석시작 모델

    # if last_round > 1:
    #     await run_ommited_device_model(template_id=WORKFLOW_TEMPLATEID, round=last_round)
    #     # update round
    #     await update_ommited_device_round(
    #         cur_round=last_round,
    #         update_round=last_round + 1,
    #         device_ids=[
    #             "UPOD130BpjBV2zbVcKxp",
    #             "-PPE5X0BpjBV2zbVwqxy",
    #             "uPO_5X0BpjBV2zbV4Kxk",
    #             "-fN8130BpjBV2zbVzasW",
    #             "7vN7130BpjBV2zbV9qs6",
    #             "APN9130BpjBV2zbVVazQ",
    #             "VvOD130BpjBV2zbV5ayU",
    #             "_vO9ZX4BpjBV2zbVwq0q",
    #             "bPOF130BpjBV2zbVk6xM",
    #             "yPPB5X0BpjBV2zbVGKzo",
    #             "NfPJ5X0BpjBV2zbVaq0F",
    #             "6fPD5X0BpjBV2zbVnax2",
    #         ],
    #     )

    # 10/24~ 분석시작 모델
    if last_round > 2:
        await run_ommited_device_model(
            template_id=WORKFLOW_TEMPLATEID,
            round=last_round - 2,
            dir_path=os.path.join(os.getenv("__DIR"), "models", "etc"),
            except_device_ids=["WfOE130BpjBV2zbVIKwr", "s_N3130BpjBV2zbVdau9", "IvPH5X0BpjBV2zbV9q3e", "y_N5130BpjBV2zbVSquL"],
        )
        await update_ommited_device_round(
            cur_round=last_round - 2,
            update_round=last_round + 1,
            device_ids=["PfPK5X0BpjBV2zbVBq1I", "__O9ZX4BpjBV2zbV1a2z", "n_OJ130BpjBV2zbVd6x9", "UfOD130BpjBV2zbVg6zu"],
        )

    # Close DB Connection
    await aliot_prophet.close_db_connection()


# run model
if __name__ == "__main__":
    asyncio.run(main())
