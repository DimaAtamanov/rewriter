import shutil

from os.path import abspath
from os import remove
from io import BytesIO
from pathlib import Path
from dotenv import dotenv_values

from requests import post, get
from uuid import uuid4

from math import ceil
from trafilatura import fetch_url, extract

from langchain.chains.llm import LLMChain
from langchain.prompts import PromptTemplate
from langchain_community.llms.gigachat import GigaChat
from PIL import Image

import pandas as pd
from pandas import DataFrame


def read_file(path: str | Path, num_articles: int | None = None) -> DataFrame:
    """Чтение табличного файла. Поддерживаемые форматы .csv, .xls, .xlsx, .xlsm, .xlsb, .odf, .ods, .odt.

    Args:
        path (str | Path): Путь к файлу
        num_articles (int | None, optional): Количество строк в DataFrame. Если None, обрабатываются все строки. Defaults to None.

    Raises:
        ValueError: Указанный формат не поддерживается

    Returns:
        Объект DataFrame
    """
    suffix = Path(path).suffix
    if suffix == ".csv":
        df = pd.read_csv(path)
    elif suffix in [".xls", ".xlsx", ".xlsm", ".xlsb", ".odf", ".ods", ".odt"]:
        df = pd.read_excel(path)
    else:
        raise ValueError(
            "Неверный формат файла. Функция поддерживает форматы .csv, .xls, .xlsx, .xlsm, .xlsb, .odf, .ods, .odt"
        )
    if num_articles:
        return df.head(num_articles)
    return df


def get_key(api_token: str, **kwargs) -> str:
    """Возвращает временный access_token для обращения к модели.

    Args:
        api_token (str): API ключ проекта

    Returns:
        str: access_token
    """
    url = "https://ngw.devices.sberbank.ru:9443/api/v2/oauth"
    rand_uuid = uuid4().__str__()
    data_encode = "scope=GIGACHAT_API_PERS"
    headers = {
        "Authorization": f"Basic {api_token}",
        "RqUID": rand_uuid,
        "Content-Type": "application/x-www-form-urlencoded",
    }
    response = post(url=url, headers=headers, data=data_encode, **kwargs)
    response.raise_for_status()

    return response.json()["access_token"]


def parse_url(url: str) -> str:
    """Возвращает текст статьи, расположенной по ссылке url.

    Args:
        url (str): адрес статьи

    Returns:
        str: текст статьи
    """
    html = fetch_url(url)
    text = extract(html, output_format="txt")

    return text


def batch_article(article: str) -> list[str]:
    """Разбивает текст на части

    Args:
        article (str): текст статьи

    Returns:
        list[str]: Список частей текста
    """
    split_text = article.split("\n")
    batch_text = []
    batch = []
    count = 0
    num_batches = ceil(len(article) / 3000)
    batch_size = ceil(len(split_text) / num_batches)

    for i, rows in enumerate(split_text):
        count += 1
        if count == batch_size:
            if len(rows) > 100:
                batch.append(rows)
            batch_text.append("\n".join(batch))
            batch = []
            count = 0
            if len(rows) < 100:
                batch.append(rows)
        else:
            batch.append(rows)
        if i == len(split_text) - 1:
            batch_text.append("\n".join(batch))

    return batch_text


def rewrite_article(model: GigaChat, batch_text: list[str]) -> str:
    markdown_template = """
    
    # Заголовок первого уровня
    
    Текст первого уровня
    
    ## Заголовок второго уровня
    
    Текст воторого уровня
    
    ### Заголовок третьего уровня
    
    Текст третьего уровня
    
    """
    rewrite_template = """

    Ты опытный копирайтер и SEO специалист с большим стажем.
    Измени текст, чтобы он стал уникальным.
    Пиши развернуто и подробно, используй сложные речевые обороты.
    Заменяй слова в исходном тексте на синонимы, меняй структуру предложений в исходном тексте.
    Измени каждый абзац исходного текста.
    Разбей свой текст на части по смыслу и придумай заголоки к этим частям.

    Статья: {article}
    
    Твой текст должен соответствовать шаблону {markdown_template}.

    """

    rewrite_promt = PromptTemplate(
        input_variables=["article", "markdown_template"], template=rewrite_template
    )
    rewrite_chain = LLMChain(prompt=rewrite_promt, llm=model)

    rewrite_batches = []
    for batch in batch_text:
        res = rewrite_chain.invoke(
            {"article": batch, "markdown_template": markdown_template}
        )
        rewrite_batches.append(res["text"])

    return "\n".join(rewrite_batches)


def make_list_paragaphs(text: str) -> list[str]:
    """Выделяет параграфы из текста на основе заголовков первого и второго уровня.

    Args:
        text (str): текст статьи

    Returns:
        list[str]: Список параграфов
    """
    split_text = [rows for rows in text.split("\n") if len(rows) != 0]
    paragraphs = []
    i = 0

    while i < len(split_text):
        if len(split_text[i]) < 100 and "#" in split_text[i]:
            j = i + 1
            paragraph = []
            while len(split_text[j]) > 100 or split_text[j].count("#") > 2:
                paragraph.append(split_text[j])
                j += 1
                if j == len(split_text):
                    break
            if len(paragraph) > 0:
                paragraphs.append("\n".join(paragraph))
            i = j
            continue
        i += 1

    return paragraphs


def summary(model: GigaChat, paragraphs: list[str]) -> list[str]:
    summary_tamplate = """

    Выдели 3 главные мысли из этого текста.
    Сформулируй каждую мысль в виде короткого предложения.

    Текст: {paragraph}

    """
    summary_promt = PromptTemplate(
        input_variables=["paragraph"], template=summary_tamplate
    )
    summary_chain = LLMChain(prompt=summary_promt, llm=model)

    list_of_summary = []
    for paragraph in paragraphs:
        if len(paragraph) < 150:
            continue
        summary = summary_chain.invoke({"paragraph": paragraph})["text"]
        list_of_summary.append(summary)

    return list_of_summary


def img_for_article(
    model: GigaChat, list_of_summary: list[str], token: str, **kwargs
) -> list:
    img_template = """

    Сгенерируй изображение на основе текста.

    Текст: {summary}

    """
    img_promt = PromptTemplate(input_variables=["summary"], template=img_template)
    img_chain = LLMChain(prompt=img_promt, llm=model)
    list_of_img = []

    for summary in list_of_summary:
        res = img_chain.invoke({"summary": summary})["text"]
        file_id = str(res[res.find("=") + 2 : res.rfind(" ") - 1])

        giga_url = (
            f"https://gigachat.devices.sberbank.ru/api/v1/files/{file_id}/content"
        )

        payload = {}
        headers = {"Accept": "application/jpg", "Authorization": f"Bearer {token}"}

        response = get(giga_url, headers=headers, data=payload, **kwargs)
        response.raise_for_status()

        io_img = BytesIO(response.content)
        img = Image.open(io_img)
        list_of_img.append(img)

    return list_of_img


def main(
    path: str,
    num_articles: int | None = None,
    with_img: bool | None = False,
    verify_ssl_certs: bool | None = None,
    verify: bool | None = None,
) -> None:
    """Основная функция. Обрабатывает ссылки во входном файле. Генерирует статьи и иллюстрации к ним

    Args:
        path (str): Путь к табличному файлу. Поддерживаемые форматы .csv, .xls, .xlsx, .xlsm, .xlsb, .odf, .ods, .odt.
        num_articles (int | None, optional): Количество строк для обработки. Если None, обрабатываются все строки. Defaults to None.
        with_img (bool | None, optional): Если True - изображения генерируются. Defaults to False.
        verify_ssl_certs (bool | None, optional): Проверка сертификата для модели. Defaults to None.
        verify (bool | None, optional): Проверка сертификата для запросов. Defaults to None.
    """
    path_env = Path(abspath(__file__)).parent.joinpath(".env")
    vars_env = dotenv_values(path_env)
    GIGACHAT_API_PERS = vars_env["GIGACHAT_API_PERS"]

    urls = read_file(path, num_articles=num_articles).iloc[:, 0].to_list()
    rewrite_text_path = Path(abspath(__file__)).parent.joinpath("rewrite_result.md")
    img_dir = Path(abspath(__file__)).parent.joinpath("img_for_article")

    if rewrite_text_path.exists():
        remove(rewrite_text_path)
    if img_dir.exists():
        shutil.rmtree(img_dir)

    for url in urls:
        print(f"\nТекущая ссылка в обработке: {url}")
        print(f"Запрашиваю ключ доступа")
        TEMP_KEY = get_key(GIGACHAT_API_PERS, verify=verify)

        print(f"Инициализирую модель")
        giga = GigaChat(access_token=TEMP_KEY, verify_ssl_certs=verify_ssl_certs)

        print(f"Извлекаю текст")
        text_article = parse_url(url)
        batch_text = batch_article(text_article)

        print(f"Провожу рерайтинг. Это может занять некоторое время")
        rewrite_text = rewrite_article(model=giga, batch_text=batch_text)

        print(f"Записываю результат рерайтинга в файл")
        with open(rewrite_text_path, "a") as file:
            file.write(rewrite_text)

        if with_img:
            list_of_paragraphs = make_list_paragaphs(rewrite_text)

            print(f"Выделяю основные темы. Это может занять некоторое время")
            list_of_summary = summary(model=giga, paragraphs=list_of_paragraphs)
            name = url.split("/")[-1] if url[-1] != "/" else url.split("/")[-2]
            curr_dir = img_dir.joinpath(name)
            curr_dir.mkdir(parents=True, exist_ok=True)

            print(f"Провожу генерацию изображений. Это может занять некоторое время")
            imgs = img_for_article(
                model=giga,
                list_of_summary=list_of_summary,
                token=TEMP_KEY,
                verify=verify,
            )
            print(f"Сохраняю изображения")
            for i, img in enumerate(imgs):
                img.save(curr_dir.joinpath(str(i) + ".jpeg"))

            with open(rewrite_text_path, "a") as file:
                file.write(f"\n\nИзображения к статье находятся в папке {curr_dir}")

        with open(rewrite_text_path, "a") as file:
            file.write("\n\n" + "*" * 100 + "\n\n" + "*" * 100 + "\n\n")

        print("\n")


if __name__ == "__main__":
    print(
        "\nВНИМАНИЕ! Убедитесь, что сохранили полученную ранее информацию. Скрипт перезаписывает файлы с результатами выполнения"
    )

    file_path = input(
        """\nУкажите полный путь к .csv или excel файлу. 
Для корректной работы столбец со ссылками должен быть САМЫМ ПЕРВЫМ по порядку.
ПЕРВАЯ СТРОКА должна содержать ЗАГОЛОВКИ.\n"""
    )

    while True:
        question = input("Обработать все строки? (y/n)\n").casefold()
        if question in ["y", "n"]:
            break
        else:
            print("Некорректный ввод. Попробуйте еще раз")

    if question == "n":
        while True:
            num_articles = input("Введите количество строк для обработки\n")
            if num_articles.isdigit() and num_articles[0] != "0":
                num_articles = int(num_articles)
                break
            else:
                print("Некорректный ввод. Попробуйте еще раз")
    else:
        num_articles = None

    while True:
        question = input("Генерировать изображения? (y/n)\n").casefold()
        if question in ["y", "n"]:
            break
        else:
            print("Некорректный ввод. Попробуйте еще раз")

    if question == "y":
        with_img = True
    else:
        with_img = False

    main(
        path=file_path,
        num_articles=num_articles,
        with_img=with_img,
    )
