import logging
import os
import re

import numpy as np
import pandas as pd
import requests
from tqdm import tqdm
from utils import Database, Singleton, get_best_match

logging.basicConfig(level=logging.INFO)
logging_name = "databases"
log = logging.getLogger(logging_name)


################################################################################
#
#                                   Databases
#
################################################################################


# @profile()
class NCBI(Database, metaclass=Singleton):
    """
    NCBI reference class
    Contains info about genes
    For mapping all protein names and aliases to the common nomenclature of official symbols of the NCBI gene database
    """

    def __init__(self, update=None):

        Database.__init__(
            self,
            update=update,
            license="Public Domain",
            license_url="https://www.ncbi.nlm.nih.gov/home/about/policies/#copyright",
        )

        self.__geneinfo = self._add_file(
            url="https://ftp.ncbi.nlm.nih.gov/gene/DATA/GENE_INFO/Mammalia/Homo_sapiens.gene_info.gz",
            update=update,
        )

        self.__geneinfo.content = self.__geneinfo.content.query(
            "Full_name_from_nomenclature_authority != '-'"
        ).drop_duplicates(
            subset="Symbol_from_nomenclature_authority",
            keep=False,
            ignore_index=True,
        )  # keep only unique official symbols
        # NCBI official symbols
        self.__ncbi_symbols = set(
            self.__geneinfo()["Symbol_from_nomenclature_authority"]
        )
        # NCBI official ids
        self.__ncbi_ids = set(self.__geneinfo()["GeneID"].astype({"GeneID": int}))

        # NCBI unique synonyms (not ambiguous)
        self.__unique_synonyms = {}
        shared_synonyms = {}
        for symbol, synonyms, ncbi_symbol in self.__geneinfo()[
            ["Symbol_from_nomenclature_authority", "Synonyms", "Symbol"]
        ].itertuples(index=False):
            if synonyms != "-":
                for synonym in synonyms.split("|") + [ncbi_symbol]:
                    # ambiguity check
                    if synonym in shared_synonyms:
                        shared_synonyms[synonym].append(symbol)
                    elif synonym in self.__unique_synonyms:
                        shared_synonyms[synonym] = [
                            self.__unique_synonyms[synonym],
                            symbol,
                        ]
                        del self.__unique_synonyms[synonym]
                    else:
                        self.__unique_synonyms[synonym] = symbol

        self.__gene2name = self.__geneinfo()[
            [
                "Symbol_from_nomenclature_authority",
                "Full_name_from_nomenclature_authority",
            ]
        ].rename(
            columns={
                "Symbol_from_nomenclature_authority": "geneSymbol",
                "Full_name_from_nomenclature_authority": "geneName",
            }
        )
        self.__gene2name_asdict = self.__gene2name.set_index("geneSymbol")[
            "geneName"
        ].to_dict()
        self.__id2symbol = (
            self.__geneinfo()[["GeneID", "Symbol_from_nomenclature_authority"]]
            .rename(
                columns={
                    "GeneID": "geneId",
                    "Symbol_from_nomenclature_authority": "geneSymbol",
                }
            )
            .astype({"geneId": int})
        )
        self.__id2symbol_asdict = self.__id2symbol.set_index("geneId")[
            "geneSymbol"
        ].to_dict()

        self.__symbol2id = (
            self.__geneinfo()[["Symbol_from_nomenclature_authority", "GeneID"]]
            .rename(
                columns={
                    "Symbol_from_nomenclature_authority": "geneSymbol",
                    "GeneID": "geneId",
                }
            )
            .astype({"geneId": int})
        )
        self.__symbol2id_asdict = self.__symbol2id.set_index("geneSymbol")[
            "geneId"
        ].to_dict()

        self.__omim2ncbi = set()
        self.__hgnc2ncbi = set()
        self.__ensembl2ncbi = set()
        for symbol, xref in self.__geneinfo()[
            ["Symbol_from_nomenclature_authority", "dbXrefs"]
        ].itertuples(index=False):
            if xref != "-":
                for key, value in {
                    id.split(":", 1)[0]: id.split(":", 1)[1] for id in xref.split("|")
                }.items():
                    if key == "MIM":
                        self.__omim2ncbi.add((value, symbol))
                    elif key == "HGNC":
                        self.__hgnc2ncbi.add((value, symbol))
                    elif key == "Ensembl":
                        self.__ensembl2ncbi.add((value, symbol))
        self.__omim2ncbi = pd.DataFrame(self.__omim2ncbi, columns=["OMIM", "NCBI"])
        self.__omim2ncbi = self.__omim2ncbi[
            ~(
                self.__omim2ncbi["OMIM"].duplicated(keep=False)
                + self.__omim2ncbi["NCBI"].duplicated(keep=False)
            )
        ]
        self.__omim2ncbi_asdict = self.__omim2ncbi.set_index("OMIM")["NCBI"].to_dict()
        self.__hgnc2ncbi = pd.DataFrame(self.__hgnc2ncbi, columns=["HGNC", "NCBI"])
        self.__hgnc2ncbi = self.__hgnc2ncbi[
            ~(
                self.__hgnc2ncbi["HGNC"].duplicated(keep=False)
                + self.__hgnc2ncbi["NCBI"].duplicated(keep=False)
            )
        ]
        self.__hgnc2ncbi_asdict = self.__hgnc2ncbi.set_index("HGNC")["NCBI"].to_dict()
        self.__ensembl2ncbi = pd.DataFrame(
            self.__ensembl2ncbi, columns=["Ensembl", "NCBI"]
        )
        self.__ensembl2ncbi = self.__ensembl2ncbi[
            ~(
                self.__ensembl2ncbi["Ensembl"].duplicated(keep=False)
                + self.__ensembl2ncbi["NCBI"].duplicated(keep=False)
            )
        ]
        self.__ensembl2ncbi_asdict = self.__ensembl2ncbi.set_index("Ensembl")[
            "NCBI"
        ].to_dict()

        log.info(f"{self.__class__.__name__} ready!")

    @property
    def symbols(self):
        """returns NCBI official symbols"""
        return self.__ncbi_symbols

    @property
    def database(self):
        return self.__geneinfo().copy()

    def check_symbol(self, symbol=None, aliases=[]):
        """
        Checks if symbols or aliases are ambiguous or can be mapped to unique NCBI symbol
        Returns the official NCBI symbol if there is one, False otherwise
        """
        if symbol is pd.NA or (not symbol and not aliases):
            return None
        elif symbol in self.__ncbi_symbols:
            return symbol
        elif symbol in self.__unique_synonyms:
            return self.__unique_synonyms[symbol]
        elif aliases:
            for alias in aliases:
                if alias in self.__ncbi_symbols:
                    return alias
                elif alias in self.__unique_synonyms:
                    return self.__unique_synonyms[alias]
        return None

    def get_symbol_by_id(self, id):
        """
        Checks if id in official NCBI ids
        Returns the official corresponding NCBI symbol if there is one, None otherwise
        """
        if not isinstance(id, int):
            if isinstance(id, str) and ("NCBI:" in id or "NCBIGene:" in id):
                id = id.split(":")[-1]
            try:
                id = int(id)
            except:
                return None
        if not id:
            return None
        elif id in self.__ncbi_ids:
            return self.check_symbol(self.__id2symbol_asdict.get(id))
        else:
            return None

    def get_id_by_symbol(self, symbol):
        """
        Checks if symbol in official NCBI symbols
        Returns the official corresponding NCBI id if there is one, None otherwise
        """
        if isinstance(symbol, str) and ("NCBI:" in symbol or "NCBIGene:" in symbol):
            symbol = symbol.split(":")[-1]
        if not symbol:
            return None
        elif symbol in self.__ncbi_symbols:
            return int(self.__symbol2id_asdict.get(self.check_symbol(symbol)))
        else:
            return None

    @property
    def id2symbol(self):
        return self.__id2symbol.copy()

    @property
    def symbol2id(self):
        return self.__symbol2id.copy()

    def get_name(self, symbol):
        """
        Given an official symbol returns the Full_name_from_nomenclature_authority
        """
        return self.__gene2name_asdict.get(symbol, f"{symbol} Not Found")

    @property
    def gene2name(self):
        return self.__gene2name.copy()

    @property
    def omim2ncbi(self):
        return self.__omim2ncbi.copy()

    @property
    def omim2ncbi_asdict(self):
        return self.__omim2ncbi_asdict.copy()

    def get_symbol_by_omim(self, omim):
        """
        Given a OMIM id returns the NCBI gene symbol
        """
        return self.__omim2ncbi_asdict.get(omim, f"{omim} Not Found")

    @property
    def hgnc2ncbi(self):
        return self.__hgnc2ncbi.copy()

    @property
    def hgnc2ncbi_asdict(self):
        return self.__hgnc2ncbi_asdict.copy()

    def get_symbol_by_hgnc(self, hgnc):
        """
        Given a HGNC id returns the NCBI gene symbol
        """
        return self.__hgnc2ncbi_asdict.get(hgnc, f"{hgnc} Not Found")

    @property
    def ensembl2ncbi(self):
        return self.__ensembl2ncbi.copy()

    @property
    def ensembl2ncbi_asdict(self):
        return self.__ensembl2ncbi_asdict.copy()

    def get_symbol_by_ensembl(self, ensembl):
        """
        Given a Ensembl id returns the NCBI gene symbol
        """
        return self.__ensembl2ncbi_asdict.get(ensembl, f"{ensembl} Not Found")


# @profile()


class DrugBank(Database, metaclass=Singleton):
    """
    DrugBank reference class
    Contains info about drugs
    """

    def __init__(self, update=None):
        try:
            email = os.environ["DRUGBANK_EMAIL"]
            password = os.environ["DRUGBANK_PASSWORD"]
        except KeyError:
            try:
                from dotenv import dotenv_values

                credentials = dotenv_values()
                email = credentials["DRUGBANK_EMAIL"]
                password = credentials["DRUGBANK_PASSWORD"]
            except:
                raise RuntimeError(
                    "No DrugBank credentials found, "
                    "drugs information will not be collected"
                )
        session = requests.Session()
        import base64

        session.headers.update(
            {
                "Authorization": f"Basic {base64.b64encode(f'{email}:{password}'.encode('ascii')).decode('ascii')}"
            }
        )
        Database.__init__(
            self,
            update=update,
            license="CC BY-NC 4.0",
            license_url="https://go.drugbank.com/releases/latest#full",
            registration_required=True,
            session=session,
            requirements=[NCBI],
        )

        def read_drugbank_full_database(filepath):
            # preparing namedtuples
            from collections import namedtuple

            print(filepath)

            atc_code = namedtuple(
                typename="atc_code",
                field_names=["level5", "level4", "level3", "level2", "level1"],
            )
            atc_code_level = namedtuple(
                typename="atc_code_level", field_names=["code", "name"]
            )
            category = namedtuple(typename="category", field_names=["name", "mesh_id"])
            interacting_drug = namedtuple(
                typename="interacting_drug",
                field_names=["name", "drugbank_id", "description"],
            )
            experimental_property = namedtuple(
                typename="experimental_property",
                field_names=["name", "value", "source"],
            )
            calculated_property = namedtuple(
                typename="calculated_property", field_names=["name", "value", "source"]
            )
            external_identifiers = namedtuple(
                typename="external_identifiers", field_names=["resource", "id"]
            )
            pathway = namedtuple(
                typename="pathway", field_names=["name", "smpdb_id", "category"]
            )
            protein_function = namedtuple(
                typename="protein_function", field_names=["general", "specific"]
            )
            protein = namedtuple(
                typename="protein",
                field_names=[
                    "drug_actions",
                    "cellular_location",
                    "chromosome_location",
                    "function",
                    "id",
                    "name",
                    "organism",
                    "swiss_prot_id",
                    "symbol",
                    "synonyms",
                    "type",
                ],
            )
            biological_entity = namedtuple(
                typename="biological_entity",
                field_names=["drug_actions", "id", "name", "organism", "type"],
            )
            small_molecule = namedtuple(
                typename="small_molecule",
                field_names=[
                    "affected_organisms",
                    "atc_codes",
                    "calculated_properties",
                    "carriers",
                    "cas_number",
                    "categories",
                    "combined_ingredients",  # ingredients of approved mixture products
                    "description",
                    "drug_interactions",
                    "enzymes",
                    "experimental_properties",
                    "external_identifiers",
                    "groups",
                    "id",
                    "indication",
                    "mechanism_of_action",
                    "name",
                    "pathways",
                    "pharmacodynamics",
                    "targets",
                    "toxicity",
                    "transporters",
                    "type",
                ],
            )

            biotech = namedtuple(
                typename="biotech",
                field_names=[
                    "affected_organisms",
                    "atc_codes",
                    "carriers",
                    "cas_number",
                    "categories",
                    "combined_ingredients",  # ingredients of approved mixture products
                    "description",
                    "drug_interactions",
                    "enzymes",
                    "experimental_properties",
                    "external_identifiers",
                    "groups",
                    "id",
                    "indication",
                    "mechanism_of_action",
                    "name",
                    "pathways",
                    "pharmacodynamics",
                    "targets",
                    "toxicity",
                    "transporters",
                    "type",
                ],
            )

            # parsing database
            import zipfile

            with zipfile.ZipFile(filepath) as z:
                with z.open(z.filelist[0].filename) as f:
                    from lxml import objectify

                    drugbank_database = objectify.parse(f).getroot().drug
            drugs_namedtuple = namedtuple(
                typename="drugs",
                field_names=tuple(d["drugbank-id"] for d in drugbank_database),
            )
            drugs = drugs_namedtuple(
                *[
                    small_molecule(
                        tuple(
                            organism.text
                            for organism in d["affected-organisms"].getchildren()
                        ),
                        tuple(
                            atc_code(
                                *[
                                    atc_code_level(
                                        codes.values()[0], f"Substance level: {d.name}"
                                    )
                                ]
                                + [
                                    atc_code_level(code.values()[0], code.text)
                                    for code in codes.iterchildren()
                                ]
                            )
                            for codes in d["atc-codes"].iterchildren()
                        ),
                        tuple(
                            calculated_property(
                                str(prop.kind), str(prop.value), str(prop.source)
                            )
                            for prop in d["calculated-properties"].getchildren()
                        ),
                        tuple(
                            protein(
                                tuple(
                                    str(action)
                                    for action in carrier.actions.getchildren()
                                ),
                                str(carrier.polypeptide["cellular-location"]),
                                str(carrier.polypeptide["chromosome-location"]),
                                protein_function(
                                    str(carrier.polypeptide["general-function"]),
                                    str(carrier.polypeptide["specific-function"]),
                                ),
                                str(carrier.id),
                                str(carrier.name),
                                str(carrier.organism),
                                str(carrier.polypeptide.values()[0]),
                                self.__NCBI.check_symbol(
                                    str(carrier.polypeptide["gene-name"])
                                ),
                                tuple(
                                    str(syn)
                                    for syn in carrier.polypeptide.synonyms.getchildren()
                                ),
                                "protein",
                            )
                            if hasattr(carrier, "polypeptide")
                            else biological_entity(
                                tuple(
                                    str(action.action)
                                    for action in carrier.actions
                                    if hasattr(action, "action")
                                ),
                                str(carrier.id),
                                str(carrier.name),
                                str(carrier.organism),
                                "biological_entity",
                            )
                            for carrier in d.carriers.getchildren()
                        ),
                        str(d["cas-number"]),
                        tuple(
                            category(str(cat.category), str(cat["mesh-id"]))
                            for cat in d.categories.iterchildren()
                        ),
                        tuple(
                            sorted(
                                {
                                    ingredient
                                    for mixture in d.mixtures.iterchildren()
                                    for ingredient in str(mixture.ingredients).split(
                                        " + "
                                    )
                                    if ingredient != d.name
                                }
                            )
                        ),
                        str(d.description),
                        tuple(
                            interacting_drug(
                                str(interaction.name),
                                str(interaction["drugbank-id"]),
                                str(interaction.description),
                            )
                            for interaction in d["drug-interactions"].getchildren()
                        ),
                        tuple(
                            protein(
                                tuple(
                                    str(action)
                                    for action in enzyme.actions.getchildren()
                                ),
                                str(enzyme.polypeptide["cellular-location"]),
                                str(enzyme.polypeptide["chromosome-location"]),
                                protein_function(
                                    str(enzyme.polypeptide["general-function"]),
                                    str(enzyme.polypeptide["specific-function"]),
                                ),
                                str(enzyme.id),
                                str(enzyme.name),
                                str(enzyme.organism),
                                str(enzyme.polypeptide.values()[0]),
                                self.__NCBI.check_symbol(
                                    str(enzyme.polypeptide["gene-name"])
                                ),
                                tuple(
                                    str(syn)
                                    for syn in enzyme.polypeptide.synonyms.getchildren()
                                ),
                                "protein",
                            )
                            if hasattr(enzyme, "polypeptide")
                            else biological_entity(
                                tuple(
                                    str(action.action)
                                    for action in enzyme.actions
                                    if hasattr(action, "action")
                                ),
                                str(enzyme.id),
                                str(enzyme.name),
                                str(enzyme.organism),
                                "biological_entity",
                            )
                            for enzyme in d.enzymes.getchildren()
                        ),
                        tuple(
                            experimental_property(
                                str(prop.kind), str(prop.value), str(prop.source)
                            )
                            for prop in d["experimental-properties"].getchildren()
                        ),
                        tuple(
                            external_identifiers(str(xid.resource), str(xid.identifier))
                            for xid in d["external-identifiers"].getchildren()
                        ),
                        tuple(str(group) for group in d.groups.getchildren()),
                        str(d["drugbank-id"]),
                        str(d["indication"]),
                        str(d["mechanism-of-action"]),
                        str(d.name),
                        tuple(
                            pathway(str(p.name), str(p["smpdb-id"]), str(p.category))
                            for p in d.pathways.getchildren()
                        ),
                        str(d.pharmacodynamics),
                        tuple(
                            protein(
                                tuple(
                                    str(action)
                                    for action in target.actions.getchildren()
                                ),
                                str(target.polypeptide["cellular-location"]),
                                str(target.polypeptide["chromosome-location"]),
                                protein_function(
                                    str(target.polypeptide["general-function"]),
                                    str(target.polypeptide["specific-function"]),
                                ),
                                str(target.id),
                                str(target.name),
                                str(target.organism),
                                str(target.polypeptide.values()[0]),
                                self.__NCBI.check_symbol(
                                    str(target.polypeptide["gene-name"])
                                ),
                                tuple(
                                    str(syn)
                                    for syn in target.polypeptide.synonyms.getchildren()
                                ),
                                "protein",
                            )
                            if hasattr(target, "polypeptide")
                            else biological_entity(
                                tuple(
                                    str(action.action)
                                    for action in target.actions
                                    if hasattr(action, "action")
                                ),
                                str(target.id),
                                str(target.name),
                                str(target.organism),
                                "biological_entity",
                            )
                            for target in d.targets.getchildren()
                        ),
                        str(d.toxicity),
                        tuple(
                            protein(
                                tuple(
                                    str(action)
                                    for action in transporter.actions.getchildren()
                                ),
                                str(transporter.polypeptide["cellular-location"]),
                                str(transporter.polypeptide["chromosome-location"]),
                                protein_function(
                                    str(transporter.polypeptide["general-function"]),
                                    str(transporter.polypeptide["specific-function"]),
                                ),
                                str(transporter.id),
                                str(transporter.name),
                                str(transporter.organism),
                                str(transporter.polypeptide.values()[0]),
                                self.__NCBI.check_symbol(
                                    str(transporter.polypeptide["gene-name"])
                                ),
                                tuple(
                                    str(syn)
                                    for syn in transporter.polypeptide.synonyms.getchildren()
                                ),
                                "protein",
                            )
                            if hasattr(transporter, "polypeptide")
                            else biological_entity(
                                tuple(
                                    str(action.action)
                                    for action in transporter.actions
                                    if hasattr(action, "action")
                                ),
                                str(transporter.id),
                                str(transporter.name),
                                str(transporter.organism),
                                "biological_entity",
                            )
                            for transporter in d.transporters.getchildren()
                        ),
                        "small_molecule",
                    )
                    if d.values()[0] == "small molecule"
                    else biotech(
                        tuple(
                            organism.text
                            for organism in d["affected-organisms"].getchildren()
                        ),
                        tuple(
                            atc_code(
                                *[
                                    atc_code_level(
                                        codes.values()[0], f"Substance level: {d.name}"
                                    )
                                ]
                                + [
                                    atc_code_level(code.values()[0], code.text)
                                    for code in codes.iterchildren()
                                ]
                            )
                            for codes in d["atc-codes"].iterchildren()
                        ),
                        tuple(
                            protein(
                                tuple(
                                    str(action)
                                    for action in carrier.actions.getchildren()
                                ),
                                str(carrier.polypeptide["cellular-location"]),
                                str(carrier.polypeptide["chromosome-location"]),
                                protein_function(
                                    str(carrier.polypeptide["general-function"]),
                                    str(carrier.polypeptide["specific-function"]),
                                ),
                                str(carrier.id),
                                str(carrier.name),
                                str(carrier.organism),
                                str(carrier.polypeptide.values()[0]),
                                self.__NCBI.check_symbol(
                                    str(carrier.polypeptide["gene-name"])
                                ),
                                tuple(
                                    str(syn)
                                    for syn in carrier.polypeptide.synonyms.getchildren()
                                ),
                                "protein",
                            )
                            if hasattr(carrier, "polypeptide")
                            else biological_entity(
                                tuple(
                                    str(action.action)
                                    for action in carrier.actions
                                    if hasattr(action, "action")
                                ),
                                str(carrier.id),
                                str(carrier.name),
                                str(carrier.organism),
                                "biological_entity",
                            )
                            for carrier in d.carriers.getchildren()
                        ),
                        str(d["cas-number"]),
                        tuple(
                            category(str(cat.category), str(cat["mesh-id"]))
                            for cat in d.categories.iterchildren()
                        ),
                        tuple(
                            sorted(
                                {
                                    ingredient
                                    for mixture in d.mixtures.iterchildren()
                                    for ingredient in str(mixture.ingredients).split(
                                        " + "
                                    )
                                    if ingredient != d.name
                                }
                            )
                        ),
                        str(d.description),
                        tuple(
                            interacting_drug(
                                str(interaction.name),
                                str(interaction["drugbank-id"]),
                                str(interaction.description),
                            )
                            for interaction in d["drug-interactions"].getchildren()
                        ),
                        tuple(
                            protein(
                                tuple(
                                    str(action)
                                    for action in enzyme.actions.getchildren()
                                ),
                                str(enzyme.polypeptide["cellular-location"]),
                                str(enzyme.polypeptide["chromosome-location"]),
                                protein_function(
                                    str(enzyme.polypeptide["general-function"]),
                                    str(enzyme.polypeptide["specific-function"]),
                                ),
                                str(enzyme.id),
                                str(enzyme.name),
                                str(enzyme.organism),
                                str(enzyme.polypeptide.values()[0]),
                                self.__NCBI.check_symbol(
                                    str(enzyme.polypeptide["gene-name"])
                                ),
                                tuple(
                                    str(syn)
                                    for syn in enzyme.polypeptide.synonyms.getchildren()
                                ),
                                "protein",
                            )
                            if hasattr(enzyme, "polypeptide")
                            else biological_entity(
                                tuple(
                                    str(action.action)
                                    for action in enzyme.actions
                                    if hasattr(action, "action")
                                ),
                                str(enzyme.id),
                                str(enzyme.name),
                                str(enzyme.organism),
                                "biological_entity",
                            )
                            for enzyme in d.enzymes.getchildren()
                        ),
                        tuple(
                            experimental_property(
                                str(prop.kind), str(prop.value), str(prop.source)
                            )
                            for prop in d["experimental-properties"].getchildren()
                        ),
                        tuple(
                            external_identifiers(str(xid.resource), str(xid.identifier))
                            for xid in d["external-identifiers"].getchildren()
                        ),
                        tuple(str(group) for group in d.groups.getchildren()),
                        str(d["drugbank-id"]),
                        str(d["indication"]),
                        str(d["mechanism-of-action"]),
                        str(d.name),
                        tuple(
                            pathway(str(p.name), str(p["smpdb-id"]), str(p.category))
                            for p in d.pathways.getchildren()
                        ),
                        str(d.pharmacodynamics),
                        tuple(
                            protein(
                                tuple(
                                    str(action)
                                    for action in target.actions.getchildren()
                                ),
                                str(target.polypeptide["cellular-location"]),
                                str(target.polypeptide["chromosome-location"]),
                                protein_function(
                                    str(target.polypeptide["general-function"]),
                                    str(target.polypeptide["specific-function"]),
                                ),
                                str(target.id),
                                str(target.name),
                                str(target.organism),
                                str(target.polypeptide.values()[0]),
                                self.__NCBI.check_symbol(
                                    str(target.polypeptide["gene-name"])
                                ),
                                tuple(
                                    str(syn)
                                    for syn in target.polypeptide.synonyms.getchildren()
                                ),
                                "protein",
                            )
                            if hasattr(target, "polypeptide")
                            else biological_entity(
                                tuple(
                                    str(action.action)
                                    for action in target.actions
                                    if hasattr(action, "action")
                                ),
                                str(target.id),
                                str(target.name),
                                str(target.organism),
                                "biological_entity",
                            )
                            for target in d.targets.getchildren()
                        ),
                        str(d.toxicity),
                        tuple(
                            protein(
                                tuple(
                                    str(action)
                                    for action in transporter.actions.getchildren()
                                ),
                                str(transporter.polypeptide["cellular-location"]),
                                str(transporter.polypeptide["chromosome-location"]),
                                protein_function(
                                    str(transporter.polypeptide["general-function"]),
                                    str(transporter.polypeptide["specific-function"]),
                                ),
                                str(transporter.id),
                                str(transporter.name),
                                str(transporter.organism),
                                str(transporter.polypeptide.values()[0]),
                                self.__NCBI.check_symbol(
                                    str(transporter.polypeptide["gene-name"])
                                ),
                                tuple(
                                    str(syn)
                                    for syn in transporter.polypeptide.synonyms.getchildren()
                                ),
                                "protein",
                            )
                            if hasattr(transporter, "polypeptide")
                            else biological_entity(
                                tuple(
                                    str(action.action)
                                    for action in transporter.actions
                                    if hasattr(action, "action")
                                ),
                                str(transporter.id),
                                str(transporter.name),
                                str(transporter.organism),
                                "biological_entity",
                            )
                            for transporter in d.transporters.getchildren()
                        ),
                        "biotech",
                    )
                    for d in tqdm(
                        drugbank_database, desc="Collecting Drugs Data from DrugBank"
                    )
                ]
            )

            return drugs

        self.__db = self._add_file(
            url=self.get_current_release_url(),
            custom_read_function=read_drugbank_full_database,
            retrieved_version=self.v,  # retrieved by get_current_release_url
        )

        self.__name2id = {drug.name: drug.id for drug in self.__db()}
        self.__drugNames = {drug.name for drug in self.__db()}

        self.__inchiKeyBase2name = {
            next(
                (
                    prop.value.split("-")[0]
                    for prop in d.calculated_properties
                    if prop.name == "InChIKey"
                ),
                None,
            ): d.name
            for d in self.__db()
            if d.type == "small_molecule"
        }
        del [self.__inchiKeyBase2name[None]]
        self.__inchiKeyBase2id = {
            next(
                (
                    prop.value.split("-")[0]
                    for prop in d.calculated_properties
                    if prop.name == "InChIKey"
                ),
                None,
            ): d.id
            for d in self.__db()
            if d.type == "small_molecule"
        }
        del [self.__inchiKeyBase2id[None]]

        log.info(f"{self.__class__.__name__} ready!")

    @property
    def database(self):
        return self.__db()

    @property
    def drugs(self):
        return self.__db()

    @property
    def inchiKeyBase2id(self):
        return self.__inchiKeyBase2id

    def get_id_by_inchiKeyBase(self, inchiKeyBase):
        return self.__inchiKeyBase2id.get(inchiKeyBase)

    @property
    def inchiKeyBase2name(self):
        return self.__inchiKeyBase2name

    def get_name_by_inchiKeyBase(self, inchiKeyBase):
        return self.__inchiKeyBase2name.get(inchiKeyBase)

    def get(self, query):
        """
        Returns a namedtuple with relevant data about the requested drug

        Accepts DrugBank IDs or drug names (returns only the exact matches)
        """
        if not isinstance(query, str):
            return None
        else:
            if query.startswith("DB"):
                return getattr(self.__db(), query)
            elif query in self.__drugNames:
                return getattr(self.__db(), self.__name2id[query])
            else:
                return None

    def search(self, query):
        """
        Returns a namedtuple with relevant data about the requested drug

        Accepts DrugBank IDs or drug names (returns the best match, if relevant)
        """
        if not isinstance(query, str):
            return None
        else:
            if query.startswith("DB"):
                return getattr(self.__db(), query)
            elif query in self.__drugNames:
                return getattr(self.__db(), self.__name2id[query])
            else:
                best_match = get_best_match(query, self.__drugNames)
                return getattr(self.__db(), self.__name2id[best_match])

    def search_drug(self, query):
        """
        Alias for search
        """
        return self.search(query)

    def get_current_release_url(self):
        if not self.update:
            try:
                import json

                with open("data/sources/sources.json", "r+") as infofile:
                    sources_data = json.load(infofile)
                    filename = list(sources_data["DrugBank"]["files"].keys())[0]
                    url = sources_data["DrugBank"]["files"][filename]["URL"]
                    version = sources_data["DrugBank"]["files"][filename]["version"]
                    self.v = re.findall("(.+)\    ", version)[
                        0
                    ]  # before four spaces (tab)
                    return url
            except Exception:
                log.warning("Unable to use local copy, forcing update")
                self._update = True
                return self.get_current_release_url()
        else:
            from bs4 import BeautifulSoup

            self.v = re.findall(
                "Version ([0-9]+\.[0-9]+\.[0-9]+) ",
                BeautifulSoup(
                    requests.get("https://go.drugbank.com/releases/latest").content,
                    "html5lib",
                ).head.title.text,
            )[0]
            url = f"https://go.drugbank.com/releases/{self.v.replace('.', '-')}/downloads/all-full-database"
            if os.path.isfile("data/sources/sources.json"):
                try:
                    import json

                    with open("data/sources/sources.json", "r+") as infofile:
                        sources_data = json.load(infofile)
                        local_url = sources_data["DrugBank"]["files"][
                            list(sources_data["DrugBank"]["files"].keys())[0]
                        ]["URL"]
                    if (
                        local_url == url
                    ):  # if there is not a newer version online it doesn't update the database
                        self._update = False
                except:
                    pass
            return url


# @profile()


class LINCS(Database):  # , metaclass=Singleton
    """
    LINCS reference class
    Contains info about gene expression profiles influenced by perturbing agents
    """

    def __init__(self, update=None, base_cell_lines=[], batch_size=None):
        Database.__init__(
            self,
            update=update,
            license="Freely Available",
            license_url="https://clue.io/connectopedia/publishing_with_geo_data, https://clue.io/connectopedia/data_redistribution, https://clue.io/connectopedia/clue_access_for_profits, https://www.ncbi.nlm.nih.gov/geo/info/disclaimer.html",
            requirements=[NCBI, DrugBank],
        )
        self.__base_cell_lines = set(base_cell_lines)
        self.__batch_size = batch_size
        if self.__batch_size:
            log.info(
                "'batch_size' provided. The database will be provided as a generator of pd.DataFrames and not as a single pd.DataFrame"
            )
        if self.__base_cell_lines == set():
            log.warning(
                "No base cell lines selected, all cell lines will be loaded and it may result in an OOM error!"
            )

        def get_hrefs_and_versions():
            import re

            import requests
            from bs4 import BeautifulSoup
            from dateutil.parser import parse as parsedate

            hrefs = {}
            versions = {}
            url = "https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE70138"
            r = requests.get(url, allow_redirects=True)
            page = BeautifulSoup(r.content, "html5lib")
            supplementary_files = page.find(
                "table", attrs={"cellpadding": "2", "cellspacing": "2", "width": "600"}
            )
            hrefs["GSE70138"] = {
                f: h
                for h in [
                    td.a.get("href") for td in supplementary_files.findAll("td") if td.a
                ]
                for f in ["Level5", "cell_info", "gene_info", "pert_info", "sig_info"]
                if f"{f.replace('_', '%5F')}%5F" in h
            }
            versions["GSE70138"] = {
                k: re.findall("\%5F([0-9]{4}\%2D[0-9]{2}\%2D[0-9]{2})\%2E", href)[
                    0
                ].replace("%2D", "-")
                for k, href in hrefs["GSE70138"].items()
            }

            url = "https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE92742"
            r = requests.get(url, allow_redirects=True)
            page = BeautifulSoup(r.content, "html5lib")
            supplementary_files = page.find(
                "table", attrs={"cellpadding": "2", "cellspacing": "2", "width": "600"}
            )
            hrefs["GSE92742"] = {
                f.replace("%2E", ""): h
                for h in [
                    td.a.get("href") for td in supplementary_files.findAll("td") if td.a
                ]
                for f in [
                    "Level5",
                    "cell_info",
                    "gene_info%2E",
                    "pert_info",
                    "sig_info",
                ]
                if f.replace("_", "%5F") in h
            }
            versions["GSE92742"] = parsedate(
                page.find("td", text="Last update date").find_next_sibling().text
            ).strftime("%Y-%m-%d")

            return hrefs, versions

        hrefs, versions = get_hrefs_and_versions()
        self.__GSE70138_cell_info = self._add_file(
            url=hrefs["GSE70138"]["cell_info"],
            retrieved_version=versions["GSE70138"]["cell_info"],
            sep="\t",
            update=update,
        )
        if self.__base_cell_lines:
            self.__cell_lines = set(
                self.__GSE70138_cell_info()[
                    self.__GSE70138_cell_info()["base_cell_id"].isin(
                        self.__base_cell_lines
                    )
                ].reset_index(drop=True)["cell_id"]
            )
        else:  # in case no cell lines are selected
            self.__cell_lines = set(self.__GSE70138_cell_info()["cell_id"])

        self.__GSE70138_gene_info = self._add_file(
            url=hrefs["GSE70138"]["gene_info"],
            retrieved_version=versions["GSE70138"]["gene_info"],
            sep="\t",
            update=update,
        )
        self.__BING_genes = set(
            self.__GSE70138_gene_info()[
                self.__GSE70138_gene_info()["pr_is_bing"] == "1"
            ]["pr_gene_id"]
        )
        self.__GSE70138_pert_info = self._add_file(
            url=hrefs["GSE70138"]["pert_info"],
            retrieved_version=versions["GSE70138"]["pert_info"],
            sep="\t",
            update=update,
        )
        self.__pertid2inchi = (
            self.__GSE70138_pert_info().set_index("pert_id")["inchi_key"].to_dict()
        )
        self.__GSE70138_sig_info = self._add_file(
            url=hrefs["GSE70138"]["sig_info"],
            retrieved_version=versions["GSE70138"]["sig_info"],
            sep="\t",
            update=update,
        )
        self.__sigid2pertid = (
            self.__GSE70138_sig_info().set_index("sig_id")["pert_id"].to_dict()
        )
        self.__sigid2cell = (
            self.__GSE70138_sig_info()[["sig_id", "cell_id"]]
            .set_index("sig_id")["cell_id"]
            .to_dict()
        )
        self.__sig_perturbed_by_trt = set(
            self.__GSE70138_sig_info()[
                self.__GSE70138_sig_info()["pert_type"].isin(("trt_cp", "trt_lig"))
            ]["sig_id"].values
            # keep only compounds, peptides and other biological agents used for treatment (https://clue.io/connectopedia/perturbagen_types_and_controls)
        )
        self.__GSE92742_cell_info = self._add_file(
            url=hrefs["GSE92742"]["cell_info"],
            retrieved_version=versions["GSE92742"],
            sep="\t",
            update=update,
        )
        if self.__base_cell_lines:
            self.__cell_lines = self.__cell_lines.union(
                set(
                    self.__GSE92742_cell_info()[
                        self.__GSE92742_cell_info()["base_cell_id"].isin(
                            self.__base_cell_lines
                        )
                    ].reset_index(drop=True)["cell_id"]
                )
            )
        else:  # in case no cell lines are selected
            self.__cell_lines = self.__cell_lines.union(
                set(self.__GSE92742_cell_info()["cell_id"])
            )

        self.__GSE92742_gene_info = self._add_file(
            url=hrefs["GSE92742"]["gene_info"],
            retrieved_version=versions["GSE92742"],
            sep="\t",
            update=update,
        )
        self.__BING_genes = self.__BING_genes.union(
            set(
                self.__GSE92742_gene_info()[
                    self.__GSE92742_gene_info()["pr_is_bing"] == "1"
                ]["pr_gene_id"]
            )
        )
        self.__GSE92742_pert_info = self._add_file(
            url=hrefs["GSE92742"]["pert_info"],
            retrieved_version=versions["GSE92742"],
            sep="\t",
            update=update,
        )
        self.__pertid2inchi.update(
            self.__GSE92742_pert_info().set_index("pert_id")["inchi_key"].to_dict()
        )
        self.__GSE92742_sig_info = self._add_file(
            url=hrefs["GSE92742"]["sig_info"],
            retrieved_version=versions["GSE92742"],
            sep="\t",
            update=update,
        )
        self.__sigid2pertid.update(
            self.__GSE92742_sig_info().set_index("sig_id")["pert_id"].to_dict()
        )
        self.__sigid2cell.update(
            (
                self.__GSE92742_sig_info()[["sig_id", "cell_id"]]
                .set_index("sig_id")["cell_id"]
                .to_dict()
            )
        )
        self.__sig_perturbed_by_trt = self.__sig_perturbed_by_trt.union(
            set(
                self.__GSE92742_sig_info()[
                    self.__GSE92742_sig_info()["pert_type"].isin(("trt_cp", "trt_lig"))
                ]["sig_id"].values
            )
            # keep only compounds, peptides and other biological agents used for treatment (https://clue.io/connectopedia/perturbagen_types_and_controls)
        )
        self.__sigid2inchi = {
            k: self.__pertid2inchi[v]
            for k, v in self.__sigid2pertid.items()
            if v in self.__pertid2inchi
        }
        self.__sigid2DBid = {
            k: self.__DrugBank.inchiKeyBase2id[v.split("-")[0]]
            for k, v in self.__sigid2inchi.items()
            if v.split("-")[0] in self.__DrugBank.inchiKeyBase2id
        }

        self.__sigid2DBname = {
            k: self.__DrugBank.inchiKeyBase2name[v.split("-")[0]]
            for k, v in self.__sigid2inchi.items()
            if v.split("-")[0] in self.__DrugBank.inchiKeyBase2name
        }

        def custom_read_expression_profiles(filepath):
            filepath = filepath.rstrip(".gz")
            if not os.path.isfile(filepath):
                import gzip
                import shutil

                with gzip.open(filepath + ".gz", "rb") as infile:
                    with open(filepath, "wb") as outfile:
                        shutil.copyfileobj(infile, outfile)
            from cmapPy.pandasGEXpress.parse import parse as parse_gctx

            cols = parse_gctx(filepath, col_meta_only=True)
            cols["col_id"] = range(len(cols))
            cols["col_name"] = cols.index
            cols.set_index("col_id", inplace=True)
            cols = cols[
                cols["col_name"].map(self.__sigid2cell.get).isin(self.__cell_lines)
            ]  # filter by selected cell lines
            cols = cols[
                cols["col_name"].isin(self.__sig_perturbed_by_trt)
            ]  # keep only signatures perturbed by compounds, peptides or other biological agents used for treatment
            cols = cols[
                cols["col_name"].isin(self.__sigid2DBid)
            ]  # keep only signatures perturbed by entities available in drugbank

            if self.__batch_size != None:
                batches = (
                    np.array_split(cols.index, len(cols) // self.__batch_size)
                    if len(cols) > self.__batch_size
                    else [cols.index]
                )

                return (
                    parse_gctx(
                        filepath,
                        cidx=batch,
                    ).data_df.loc[list(self.__BING_genes)]
                    for batch in batches
                )
            else:
                return parse_gctx(
                    filepath,
                ).data_df.loc[list(self.__BING_genes)]

        self.__GSE70138_level5 = self._add_file(
            url=hrefs["GSE70138"]["Level5"],
            retrieved_version=versions["GSE70138"]["Level5"],
            custom_read_function=custom_read_expression_profiles,
            update=update,
        )
        self.__GSE92742_level5 = self._add_file(
            url=hrefs["GSE92742"]["Level5"],
            retrieved_version=versions["GSE92742"],
            custom_read_function=custom_read_expression_profiles,
            update=update,
        )

    @property
    def base_cell_lines(self):
        return self.__base_cell_lines.copy()

    @property
    def cell_lines(self):
        return self.__cell_lines.copy()

    @property
    def BING_genes(self):
        return self.__BING_genes.copy()

    @property
    def sig_perturbed_by_trt(self):
        """
        Names of signatures perturbed by compounds, peptides or other biological agents used for treatment
        """
        return self.__sig_perturbed_by_trt.copy()

    @property
    def sigid2cell(self):
        return self.__sigid2cell.copy()

    def get_cell_by_sigid(self, sigid):
        return self.__sigid2cell.get(sigid)

    @property
    def sigid2pertid(self):
        return self.__sigid2pertid.copy()

    def get_pertid_by_sigid(self, sigid):
        return self.__sigid2pertid.get(sigid)

    @property
    def sigid2DBid(self):
        return self.__sigid2DBid.copy()

    def get_DBid_by_sigid(self, sigid):
        return self.__sigid2DBid.get(sigid)

    @property
    def sigid2DBname(self):
        return self.__sigid2DBname.copy()

    def get_DBname_by_sigid(self, sigid):
        return self.__sigid2DBname.get(sigid)

    @property
    def GSE70138_cell_info(self):
        return self.__GSE70138_cell_info().copy()

    @property
    def GSE92742_cell_info(self):
        return self.__GSE92742_cell_info().copy()

    @property
    def GSE70138_gene_info(self):
        return self.__GSE70138_gene_info().copy()

    @property
    def GSE92742_gene_info(self):
        return self.__GSE92742_gene_info().copy()

    @property
    def GSE70138_pert_info(self):
        return self.__GSE70138_pert_info().copy()

    @property
    def GSE92742_pert_info(self):
        return self.__GSE92742_pert_info().copy()

    @property
    def GSE70138_sig_info(self):
        return self.__GSE70138_sig_info().copy()

    @property
    def GSE92742_sig_info(self):
        return self.__GSE92742_sig_info().copy()

    @property
    def GSE70138_level5(self):
        if self.__batch_size:
            content = self.__GSE70138_level5()
            self.__GSE70138_level5.read()
            return content
        else:
            return self.__GSE70138_level5().copy()

    @property
    def GSE92742_level5(self):
        if self.__batch_size:
            content = self.__GSE92742_level5()
            self.__GSE92742_level5.read()
            return content
        else:
            return self.__GSE92742_level5().copy()

    @property
    def database(self):
        if self.__batch_size:
            from itertools import chain

            return chain(self.GSE70138_level5, self.GSE92742_level5)
        else:
            return self.GSE70138_level5.join(self.GSE92742_level5)