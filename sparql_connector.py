from rdflib import Graph
import os, requests, json
import logging

logger = logging.getLogger(__name__)
rootFolder = os.path.dirname(os.path.realpath(__file__))
KB_ttl_path = rootFolder + "/data_store/DDHub_model/DWISVocabulary_merged.ttl"
# SPARQL_endpoint = "http://localhost:3030/DWISVocabulary/"


def query_from_KB_ttl(ttl_file_path: str, query: str) -> dict:
    g = Graph()
    g.parse(ttl_file_path, format="ttl")

    # Apply the query to the graph and iterate through results
    response = g.query(query)
    data = []
    for row in response:
        col = []
        for c in row:
            col.append(str(c))
        data.append(col)
    result = {
        "header": [var.n3()[1:] for var in response.vars],
        "data": data,
    }
    return result


def query_from_KB(endpoint: str, query: str) -> dict:
    response = requests.post(
        endpoint,
        data={"query": query},
        headers={"Accept": "application/sparql-results+json"},
    )
    response_dict = response.json()
    header = response_dict["head"]["vars"]
    # header = [var.n3()[1:] for var in response_dict["head"]["vars"]]
    data = []
    for body in response_dict["results"]["bindings"]:
        values = []
        for key in body:
            values.append(body[key]["value"])
        data.append(values)
    result = {
        "header": header,
        "data": data,
    }
    return result


def make_result_table(query_result: dict, no_namespace: bool = True) -> list:
    result_table = []
    result_data = query_result["data"]
    for r in result_data:
        print(r)
    return result_table


def parse_keywords(input_string):
    keywords = input_string.split("?")[1:]
    keywords = [keyword.strip() for keyword in keywords]
    result = {keyword: index for index, keyword in enumerate(keywords)}
    return result


def separate_list_item(input_list: list) -> list:
    output_set = set()
    for item in input_list:
        # 按逗号分隔并去除空格
        list_tmp = item.split(",")
        output_set.update(list_tmp)
    return list(output_set)


"""
To add new SPARQL query module:
1. Write a "make_sparql_xxx()" function to generate SPARQL string.
2. Write a controlling function to cover: SPARQL string generation, query_from_KB, result extraction.
"""


def make_querytring_MQuantity_relatedTo_ProtytypeData(PrototypeData: str) -> str:
    query_template = """
PREFIX rdf: <http://www.w3.org/1999/02/22-rdf-syntax-ns#>
PREFIX owl: <http://www.w3.org/2002/07/owl#>
PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>
PREFIX xsd: <http://www.w3.org/2001/XMLSchema#>
PREFIX ddhub: <http://ddhub.no/>
SELECT DISTINCT ?MeasurableQuantity
WHERE { 
    ddhub:<PrototypeData> rdf:type owl:Class ;
            rdfs:subClassOf [ 
                rdf:type owl:Restriction ;
                owl:onProperty ddhub:IsOfMeasurableQuantity ;
                owl:allValuesFrom ?MeasurableQuantity ].
}
    """
    query = query_template.replace("<PrototypeData>", PrototypeData)
    return query


def query_MQuantity(ttl_file_path: str, PrototypeData: str):
    query = make_querytring_MQuantity_relatedTo_ProtytypeData(PrototypeData)
    response = query_from_KB_ttl(ttl_file_path, query)
    # response = query_from_KB(SPARQL_endpoint, query)
    # print(response)
    MeasurableQuantity = response["data"][0]
    return MeasurableQuantity


def make_queryString_MQuantity_Quantity(PrototypeData: str) -> str:
    query_template = """
PREFIX rdf: <http://www.w3.org/1999/02/22-rdf-syntax-ns#>
PREFIX owl: <http://www.w3.org/2002/07/owl#>
PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>
PREFIX xsd: <http://www.w3.org/2001/XMLSchema#>
PREFIX ddhub: <http://ddhub.no/>
SELECT DISTINCT ?MeasurableQuantity ?Quantity
WHERE { 
    ddhub:<PrototypeData> rdf:type owl:Class ;
        rdfs:subClassOf [ 
                rdf:type owl:Restriction ;
                owl:onProperty ddhub:IsOfMeasurableQuantity ;
                owl:allValuesFrom ?MeasurableQuantity ].

    ?MeasurableQuantity rdf:type owl:Class ;
        rdfs:subClassOf [ 
                rdf:type owl:Restriction ;
                owl:onProperty ddhub:IsOfBaseQuantity ;
                owl:allValuesFrom ?Quantity
              ] .
}
    """
    query = query_template.replace("<PrototypeData>", PrototypeData)
    return query


def query_MQuantity_Quantity(ttl_file_path: str, PrototypeData: str):
    query = make_queryString_MQuantity_Quantity(PrototypeData)
    response = query_from_KB_ttl(ttl_file_path, query)
    # response = query_from_KB(SPARQL_endpoint, query)
    # print(response)
    # MeasurableQuantity = response["data"][0]
    # Quantity = response["data"][0]
    return response


def make_queryString_Unit_relatedTo_Quantiy(Quantity: str) -> str:
    query_template = """
PREFIX rdf: <http://www.w3.org/1999/02/22-rdf-syntax-ns#>
PREFIX owl: <http://www.w3.org/2002/07/owl#>
PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>
PREFIX xsd: <http://www.w3.org/2001/XMLSchema#>
PREFIX ddhub: <http://ddhub.no/>
SELECT DISTINCT ?Unit
WHERE {
    <query_body>
}
    """
    query_body_tmpl_1 = """
    ?Unit rdfs:subClassOf ?restriction1 .
    ?restriction1 rdf:type owl:Restriction ;
        owl:onProperty ddhub:IsUnitForQuantity ;
        owl:allValuesFrom ?allValuesClass .

    ?allValuesClass owl:unionOf ?unionList .
    ?unionList rdf:rest*/rdf:first ddhub:<Quantity> . 
"""
    query_body_tmpl_2 = """
    ?Unit rdfs:subClassOf ?restriction1 .
    ?restriction1 rdf:type owl:Restriction ;
        owl:onProperty ddhub:IsUnitForQuantity .
"""
    if Quantity != "None":
        query_body = query_body_tmpl_1.replace("<Quantity>", Quantity)
    else:
        query_body = query_body_tmpl_2
    query = query_template.replace("<query_body>", query_body)
    return query


def query_Unit_relatedTo_Quantiy(ttl_file_path: str, Quantity: str):
    query = make_queryString_Unit_relatedTo_Quantiy(Quantity)
    response = query_from_KB_ttl(ttl_file_path, query)
    # response = query_from_KB(SPARQL_endpoint, query)
    # print(response)
    unit_list = []
    for u in response["data"]:
        unit_list.append(u[0].split("/")[-1])
    return unit_list


def make_queryString_PrototypeData_relatedTo_Quantiy(Quantity: str) -> str:
    query_template = """
PREFIX rdf: <http://www.w3.org/1999/02/22-rdf-syntax-ns#>
PREFIX owl: <http://www.w3.org/2002/07/owl#>
PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>
PREFIX xsd: <http://www.w3.org/2001/XMLSchema#>
PREFIX ddhub: <http://ddhub.no/>
SELECT DISTINCT ?PrototypeData
WHERE {
    <query_body>
}
Order by ?PrototypeData
    """
    query_body_tmpl_1 = """
    ?PrototypeData rdfs:subClassOf ?restriction1 .
    ?restriction1 rdf:type owl:Restriction ;
        owl:onProperty ddhub:IsOfMeasurableQuantity ;
        owl:allValuesFrom ?mq .

    ?mq rdfs:subClassOf ?restriction2 .
    ?restriction2 rdf:type owl:Restriction ;
        owl:onProperty ddhub:IsOfBaseQuantity ;
        owl:allValuesFrom ddhub:<Quantity> .
"""
    query_body_tmpl_2 = """
    ?PrototypeData rdfs:subClassOf ?restriction1 .
    ?restriction1 rdf:type owl:Restriction ;
        owl:onProperty ddhub:IsOfMeasurableQuantity .
"""
    if Quantity != "None":
        query_body = query_body_tmpl_1.replace("<Quantity>", Quantity)
    else:
        query_body = query_body_tmpl_2
    query = query_template.replace("<query_body>", query_body)
    return query


def query_PrototypeData_relatedTo_Quantiy(ttl_file_path: str, quantity: str):
    query = make_queryString_PrototypeData_relatedTo_Quantiy(quantity)
    response = query_from_KB_ttl(ttl_file_path, query)
    # response = query_from_KB(SPARQL_endpoint, query)
    # print(response)
    prototypeData_list = []
    for u in response["data"]:
        prototypeData_list.append(u[0].split("/")[-1])
    return prototypeData_list


def make_queryStr_PrototypeData_fullList_extraContent():
    query = """
PREFIX rdf: <http://www.w3.org/1999/02/22-rdf-syntax-ns#>
PREFIX owl: <http://www.w3.org/2002/07/owl#>
PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>
PREFIX xsd: <http://www.w3.org/2001/XMLSchema#>
PREFIX ddhub: <http://ddhub.no/>
PREFIX zzz: <http://ddhub.demo/zzz#>
SELECT DISTINCT ?PrototypeData ?comment ?MeasurableQuantity ?Quantity ?commonMnemonics
WHERE {
    ?PrototypeData rdfs:subClassOf ddhub:PrototypeData.
    ?PrototypeData rdfs:comment ?comment.
    ?PrototypeData rdfs:subClassOf ?restriction .
    ?restriction rdf:type owl:Restriction ;
                owl:onProperty ddhub:IsOfMeasurableQuantity ;
                owl:allValuesFrom ?MeasurableQuantity .
    ?MeasurableQuantity rdfs:subClassOf ?restriction2 .
    ?restriction2 rdf:type owl:Restriction ;
                owl:onProperty ddhub:IsOfBaseQuantity ;
                owl:allValuesFrom ?Quantity .
    Filter (STR(?comment) != "")

    OPTIONAL { ?PrototypeData zzz:commonMnemonics ?commonMnemonics.}
}
ORDER BY ?PrototypeData
    """
    return query


def generate_PrototypeData_fullList_extraContent(source_ttl_path: str) -> dict:
    query = make_queryStr_PrototypeData_fullList_extraContent()
    response = query_from_KB_ttl(source_ttl_path, query)
    # response = query_from_KB(SPARQL_endpoint, query)
    # print(response)
    # keyIndex = parse_keywords("?PrototypeData ?comment ?MeasurableQuantity")
    fullList_extraContent = {}
    for r in response["data"]:
        PrototypeData = str(r[0]).split("/")[-1]
        comment = str(r[1])
        IsOfMeasurableQuantity = str(r[2]).split("/")[-1]
        IsOfBaseQuantity = str(r[3]).split("/")[-1]
        commonMnemonics = str(r[4]) if r[4] else "None"

        if PrototypeData in fullList_extraContent:
            # To prevent the overwritting.
            commentList = fullList_extraContent[PrototypeData]["rdfs:comment"]
            commonMnemonicsList = (
                fullList_extraContent[PrototypeData]["zzz:commonMnemonics"]
                if "zzz:commonMnemonics" in fullList_extraContent[PrototypeData]
                else ["None"]
            )
            # tmp[Quantity]["rdfs:comment"] += " " + fullList_extraContent[Quantity]["rdfs:comment"]
            if comment not in commentList:
                commentList.append(comment)
            if commonMnemonics not in commonMnemonicsList:
                commonMnemonicsList.append(commonMnemonics)
        else:
            commentList = [comment]
            commonMnemonicsList = [commonMnemonics]

        tmp = {
            PrototypeData: {
                "ddhub:PrototypeData": PrototypeData,
                "rdfs:comment": commentList,
                "ddhub:IsOfMeasurableQuantity": IsOfMeasurableQuantity,
                "ddhub:IsOfBaseQuantity": IsOfBaseQuantity,
            }
        }

        if commonMnemonicsList != [None] and commonMnemonicsList != [""] and commonMnemonicsList != ["None"]:
            tmp[PrototypeData]["zzz:commonMnemonics"] = separate_list_item(commonMnemonicsList)

        fullList_extraContent.update(tmp)
    return fullList_extraContent


def make_queryStr_Unit_fullList_extraContent():
    query = """
PREFIX rdf: <http://www.w3.org/1999/02/22-rdf-syntax-ns#>
PREFIX owl: <http://www.w3.org/2002/07/owl#>
PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>
PREFIX xsd: <http://www.w3.org/2001/XMLSchema#>
PREFIX ddhub: <http://ddhub.no/>
PREFIX zzz: <http://ddhub.demo/zzz#>
SELECT DISTINCT ?Unit ?comment ?Quantity ?commonMnemonics
WHERE {
    ?Unit rdfs:subClassOf ddhub:Unit.
    ?Unit rdfs:comment ?comment.
    ?Unit rdfs:subClassOf ?restriction .
    ?restriction rdf:type owl:Restriction ;
                owl:onProperty ddhub:IsUnitForQuantity ;
                owl:allValuesFrom ?allValuesClass .

    ?allValuesClass owl:unionOf ?unionList .
    ?unionList rdf:rest*/rdf:first ?Quantity .

    Filter (STR(?comment) != "")
    OPTIONAL { ?Unit zzz:commonMnemonics ?commonMnemonics.}
}
ORDER BY ?Unit
    """
    return query


def generate_Unit_fullList_extraContent(source_ttl_path: str) -> dict:
    query = make_queryStr_Unit_fullList_extraContent()
    response = query_from_KB_ttl(source_ttl_path, query)
    # response = query_from_KB(SPARQL_endpoint, query)
    # print(response)
    # keyIndex = parse_keywords("?PrototypeData ?comment ?MeasurableQuantity")
    fullList_extraContent = {}
    for r in response["data"]:
        Unit = str(r[0]).split("/")[-1]
        comment = str(r[1])
        IsUnitForQuantity = str(r[2]).split("/")[-1]
        commonMnemonics = str(r[3]) if r[3] else None

        if Unit in fullList_extraContent:
            # To prevent the overwritting.
            commentList = fullList_extraContent[Unit]["rdfs:comment"]
            commonMnemonicsList = (
                fullList_extraContent[Unit]["zzz:commonMnemonics"]
                if "zzz:commonMnemonics" in fullList_extraContent[Unit]
                else ["None"]
            )
            IsUnitForQuantityList = fullList_extraContent[Unit]["ddhub:IsUnitForQuantity"]
            # tmp[Quantity]["rdfs:comment"] += " " + fullList_extraContent[Quantity]["rdfs:comment"]
            if comment not in commentList:
                commentList.append(comment)
            if commonMnemonics not in commonMnemonicsList:
                commonMnemonicsList.append(commonMnemonics)
            if IsUnitForQuantity not in IsUnitForQuantityList:
                IsUnitForQuantityList.append(IsUnitForQuantity)
        else:
            commentList = [comment]
            commonMnemonicsList = [commonMnemonics]
            IsUnitForQuantityList = [IsUnitForQuantity]

        tmp = {
            Unit: {
                "ddhub:Unit": Unit,
                "rdfs:comment": commentList,
                "ddhub:IsUnitForQuantity": IsUnitForQuantityList,
            }
        }

        if commonMnemonicsList != [None] and commonMnemonicsList != [""] and commonMnemonicsList != ["None"]:
            tmp[Unit]["zzz:commonMnemonics"] = separate_list_item(commonMnemonicsList)

        fullList_extraContent.update(tmp)
    return fullList_extraContent


def make_queryStr_Quantity_fullList_extraContent():
    query = """
PREFIX rdf: <http://www.w3.org/1999/02/22-rdf-syntax-ns#>
PREFIX owl: <http://www.w3.org/2002/07/owl#>
PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>
PREFIX xsd: <http://www.w3.org/2001/XMLSchema#>
PREFIX ddhub: <http://ddhub.no/>
PREFIX zzz: <http://ddhub.demo/zzz#>
SELECT DISTINCT ?Quantity ?comment ?Unit
WHERE {
    ?Quantity rdfs:comment ?comment.
    Filter (STR(?comment) != "")

    ?Unit rdfs:subClassOf ddhub:Unit.
    ?Unit rdfs:subClassOf ?restriction .
    ?restriction rdf:type owl:Restriction ;
                owl:onProperty ddhub:IsUnitForQuantity ;
                owl:allValuesFrom ?allValuesClass .

    ?allValuesClass owl:unionOf ?unionList .
    ?unionList rdf:rest*/rdf:first ?Quantity .
}
ORDER BY ?Quantity
    """
    return query


def generate_Quantity_fullList_extraContent(source_ttl_path: str) -> dict:
    query = make_queryStr_Quantity_fullList_extraContent()
    response = query_from_KB_ttl(source_ttl_path, query)
    # response = query_from_KB(SPARQL_endpoint, query)
    # print(response)
    fullList_extraContent = {}
    for r in response["data"]:
        Quantity = str(r[0]).split("/")[-1]
        comment = str(r[1])
        QuantityHasUnit = str(r[2]).split("/")[-1]

        if Quantity in fullList_extraContent:
            # To prevent the overwritting.
            commentList = fullList_extraContent[Quantity]["rdfs:comment"]
            unitList = fullList_extraContent[Quantity]["zzz:QuantityHasUnit"]
            # tmp[Quantity]["rdfs:comment"] += " " + fullList_extraContent[Quantity]["rdfs:comment"]
            if comment not in commentList:
                commentList.append(comment)
            if QuantityHasUnit not in unitList:
                unitList.append(QuantityHasUnit)
        else:
            commentList = [comment]
            unitList = [QuantityHasUnit]

        tmp = {
            Quantity: {
                "ddhub:Quantity": Quantity,
                "rdfs:comment": commentList,
                "zzz:QuantityHasUnit": unitList,
            }
        }

        fullList_extraContent.update(tmp)
    return fullList_extraContent


# -------------


def test_ask_MeasurableQuantity():
    ttl_file_path = KB_ttl_path
    r = query_MQuantity(ttl_file_path, "HookLoad")
    print(r)


def test_retrieve_context():
    currentFolder = os.path.dirname(os.path.realpath(__file__))
    ttl_file_path = KB_ttl_path

    context = generate_PrototypeData_fullList_extraContent(ttl_file_path)
    with open(currentFolder + "/data_store/DDHub_model/PrototypeData_fullList_extraContent.json", "w") as json_file:
        json.dump(context, json_file, indent=4)


if __name__ == "__main__":
    # test_ask_MeasurableQuantity()
    test_retrieve_context()
