{
    "class": "CommandLineTool",
    "cwlVersion": "v1.2",
    "$namespaces": {
        "sbg": "https://sevenbridges.com"
    },
    "id": "yiran/variant-workbench-testing/gene-based-variant-filtering/20",
    "baseCommand": [
        "spark-submit"
    ],
    "inputs": [
        {
            "id": "gene_list",
            "type": "string",
            "inputBinding": {
                "prefix": "-g",
                "shellQuote": false,
                "position": 1
            }
        },
        {
            "id": "study_id",
            "type": "string",
            "inputBinding": {
                "prefix": "-s",
                "shellQuote": false,
                "position": 1
            }
        },
        {
            "id": "spark_driver_mem",
            "type": "int?",
            "doc": "GB of RAM to allocate to this task",
            "default": 10
        },
        {
            "loadListing": "deep_listing",
            "sbg:suggestedValue": {
                "class": "Directory",
                "path": "63c9b43d5ec27133d2fca2c9",
                "name": "hg38_HGMD2022Q4_variant"
            },
            "id": "hgmd_parquet",
            "type": "Directory",
            "inputBinding": {
                "prefix": "--hgmd",
                "shellQuote": false,
                "position": 0
            },
            "doc": "HGMD variatn parquet file dir"
        },
        {
            "loadListing": "deep_listing",
            "sbg:suggestedValue": {
                "class": "Directory",
                "path": "63779b0180315a3645400d9a",
                "name": "dbnsfp"
            },
            "id": "dbnsfp_annovar_parquet",
            "type": "Directory",
            "inputBinding": {
                "prefix": "--dbnsfp",
                "shellQuote": false,
                "position": 0
            },
            "doc": "dbnsfp annovar parquet file dir"
        },
        {
            "loadListing": "deep_listing",
            "sbg:suggestedValue": {
                "class": "Directory",
                "path": "637410ead933e90bfba2d8ea",
                "name": "clinvar"
            },
            "id": "clinvar_parquet",
            "type": "Directory",
            "inputBinding": {
                "prefix": "--clinvar",
                "shellQuote": false,
                "position": 0
            },
            "doc": "clinvar parquet file dir"
        },
        {
            "loadListing": "deep_listing",
            "sbg:suggestedValue": {
                "class": "Directory",
                "path": "63ed328e7a0654635c6a72a1",
                "name": "consequences_re_000019"
            },
            "id": "consequences_parquet",
            "type": "Directory",
            "inputBinding": {
                "prefix": "--consequences",
                "shellQuote": false,
                "position": 0
            },
            "doc": "consequences parquet file dir"
        },
        {
            "loadListing": "deep_listing",
            "sbg:suggestedValue": {
                "class": "Directory",
                "path": "63ed328e7a0654635c6a729d",
                "name": "variants_re_000019_20230212_003710"
            },
            "id": "variants_parquet",
            "type": "Directory",
            "inputBinding": {
                "prefix": "--variants",
                "shellQuote": false,
                "position": 0
            },
            "doc": "variants parquet file dir"
        },
        {
            "loadListing": "deep_listing",
            "sbg:suggestedValue": {
                "class": "Directory",
                "path": "63ff628bf897792af067fa05",
                "name": "diagnoses_re_000019"
            },
            "id": "diagnoses_parquet",
            "type": "Directory",
            "inputBinding": {
                "prefix": "--diagnoses",
                "shellQuote": false,
                "position": 0
            }
        },
        {
            "loadListing": "deep_listing",
            "sbg:suggestedValue": {
                "class": "Directory",
                "path": "63ff628ef897792af067ffd4",
                "name": "phenotypes_re_000019"
            },
            "id": "phenotypes_parquet",
            "type": "Directory",
            "inputBinding": {
                "prefix": "--phenotypes",
                "shellQuote": false,
                "position": 0
            },
            "doc": "phenotypes parquet file dir"
        },
        {
            "id": "gnomAD_TOPMed_maf",
            "type": "double",
            "inputBinding": {
                "prefix": "--maf",
                "shellQuote": false,
                "position": 0
            },
            "default": 0.0001
        },
        {
            "id": "damage_predict_count_lower",
            "type": "double?",
            "inputBinding": {
                "prefix": "--dpc_l",
                "shellQuote": false,
                "position": 0
            },
            "default": 0.5
        },
        {
            "id": "damage_predict_count_upper",
            "type": "double?",
            "inputBinding": {
                "prefix": "--dpc_u",
                "shellQuote": false,
                "position": 0
            },
            "default": 1
        },
        {
            "id": "alternative_allele_frequency",
            "type": "double?",
            "inputBinding": {
                "prefix": "--aaf",
                "shellQuote": false,
                "position": 0
            },
            "doc": "alternative allele frequency",
            "default": 0.2
        },
        {
            "id": "known_variants_l",
            "type": "string?",
            "inputBinding": {
                "prefix": "--known_variants_l",
                "shellQuote": false,
                "position": 0
            },
            "default": "ClinVar HGMD"
        },
        {
            "loadListing": "deep_listing",
            "sbg:suggestedValue": {
                "class": "Directory",
                "path": "63ff628ff897792af0680182",
                "name": "studies_re_000019"
            },
            "id": "studies_parquet",
            "type": "Directory",
            "inputBinding": {
                "prefix": "--studies",
                "shellQuote": false,
                "position": 0
            }
        },
        {
            "id": "sql_broadcastTimeout",
            "type": "int?",
            "doc": ".config(\"spark.sql.broadcastTimeout\", 36000)",
            "default": 36000
        }
    ],
    "outputs": [
        {
            "id": "output",
            "type": "File?",
            "outputBinding": {
                "glob": "*.tsv"
            }
        }
    ],
    "doc": "get a list of pathogenic and likely pathogenic variants in interested genes from specified study cohort(s).",
    "label": "Gene_Based_Variant_Filtering",
    "arguments": [
        {
            "prefix": "",
            "shellQuote": false,
            "position": 0,
            "valueFrom": "--packages io.projectglow:glow-spark3_2.12:1.1.2  --conf spark.hadoop.io.compression.codecs=io.projectglow.sql.util.BGZFCodec --conf spark.sql.broadcastTimeout=$(inputs.sql_broadcastTimeout) --driver-memory $(inputs.spark_driver_mem)G  Gene_based_variant_filtering.py"
        }
    ],
    "requirements": [
        {
            "class": "ShellCommandRequirement"
        },
        {
            "class": "LoadListingRequirement"
        },
        {
            "class": "DockerRequirement",
            "dockerPull": "pgc-images.sbgenomics.com/d3b-bixu/pyspark:3.1.2"
        },
        {
            "class": "InitialWorkDirRequirement",
            "listing": [
                {
                    "entryname": "Gene_based_variant_filtering.py",
                    "entry": "import argparse\nfrom argparse import RawTextHelpFormatter\nimport pyspark\nfrom pyspark.sql import DataFrame, functions as F\nfrom pyspark.sql.types import DoubleType\nimport glow\nfrom functools import reduce\n\nparser = argparse.ArgumentParser(\n    description='Script of gene based variant filtering. \\n\\\n    MUST BE RUN WITH spark-submit. For example: \\n\\\n    spark-submit --driver-memory 10G Gene_based_variant_filtering.py',\n    formatter_class=RawTextHelpFormatter)\n\nparser = argparse.ArgumentParser()\nparser.add_argument('-g', '--gene_symbols', nargs='+', default=[],\n                    help='list of gene that we interested')\nparser.add_argument('-s', '--study_ids', nargs='+', default=[],\n                    help='list of study to look for')\nparser.add_argument('--hgmd',\n        help='HGMD variant parquet file dir')\nparser.add_argument('--dbnsfp',\n        help='dbnsfp annovar parquet file dir')\nparser.add_argument('--clinvar',\n        help='clinvar parquet file dir')\nparser.add_argument('--consequences',\n        help='consequences parquet file dir')\nparser.add_argument('--variants',\n        help='variants parquet file dir')\nparser.add_argument('--diagnoses',\n        help='diagnoses parquet file dir')\nparser.add_argument('--phenotypes',\n        help='phenotypes parquet file dir')\nparser.add_argument('--studies',\n        help='studies parquet file dir')\nparser.add_argument('--maf',\n        help='gnomAD and TOPMed max allele frequency')\nparser.add_argument('--dpc_l',\n        help='damage predict count lower threshold')\nparser.add_argument('--dpc_u',\n        help='damage predict count upper threshold')\nparser.add_argument('--known_variants_l', nargs='+', default=['ClinVar', 'HGMD'],\n                    help='known variant databases used, default is ClinVar and HGMD')\nparser.add_argument('--aaf',\n        help='alternative allele frequency threshold')\n\nargs = parser.parse_args()\n\n# Create spark session\nspark = (\n    pyspark.sql.SparkSession.builder.appName(\"PythonPi\")\n    .getOrCreate()\n    )\n# Register so that glow functions like read vcf work with spark. Must be run in spark shell or in context described in help\nspark = glow.register(spark)\n\n# gene based variant filtering\ndef gene_based_filt(args):\n    # parameter configuration\n    gene_symbols_trunc = args.gene_symbols\n    study_id_list = args.study_ids\n    gnomAD_TOPMed_maf = args.maf\n    dpc_l = args.dpc_l\n    dpc_u = args.dpc_u\n    known_variants_l = args.known_variants_l\n    aaf = args.aaf\n\n    # customized tables loading\n    hg38_HGMD2022Q4_variant = spark.read.parquet(args.hgmd)\n    dbnsfp_annovar = spark.read.parquet(args.dbnsfp)\n    clinvar = spark.read.parquet(args.clinvar)\n    consequences = spark.read.parquet(args.consequences)\n    variants = spark.read.parquet(args.variants)\n    diagnoses = spark.read.parquet(args.diagnoses)\n    phenotypes = spark.read.parquet(args.phenotypes)\n\n    occ_dict = []\n    for s_id in study_id_list:\n        occ_dict.append(spark.read.parquet('/sbgenomics/project-files/occurrences_Yiran/occurrences_*/study_id=' + s_id))\n    occurrences = reduce(DataFrame.unionAll, occ_dict)\n\n    #  Actual running step, generating table t_output\n    cond = ['chromosome', 'start', 'reference', 'alternate']\n\n    # Table consequences, restricted to canonical annotation and input genes/study IDs\n    c_csq = ['consequences', 'rsID', 'impact', 'symbol', 'ensembl_gene_id', 'refseq_mrna_id', 'hgvsg', 'hgvsc',\n             'hgvsp', 'study_ids']\n    t_csq = consequences.withColumnRenamed('name', 'rsID') \\\n        .drop('variant_class') \\\n        .where((F.col('original_canonical') == 'true') \\\n               & (F.col('symbol').isin(gene_symbols_trunc)) \\\n               & (F.size(F.array_intersect(F.col('study_ids'), F.lit(F.array(*map(F.lit, study_id_list))))) > 0)) \\\n        .select(cond + c_csq)\n    chr_list = [c['chromosome'] for c in t_csq.select('chromosome').distinct().collect()]\n\n    # Table dbnsfp_annovar, added a column for ratio of damage predictions to all predictions\n    c_dbn = ['DamagePredCount', 'PredCountRatio_D2T', 'TWINSUK_AF', 'ALSPAC_AF', 'UK10K_AF']\n    t_dbn = dbnsfp_annovar \\\n        .where(F.col('chromosome').isin(chr_list)) \\\n        .withColumn('PredCountRatio_D2T',\n                    F.when(F.split(F.col('DamagePredCount'), '_')[1] == 0, F.lit(None).cast(DoubleType())) \\\n                    .otherwise(\n                        F.split(F.col('DamagePredCount'), '_')[0] / F.split(F.col('DamagePredCount'), '_')[1])) \\\n        .select(cond + c_dbn)\n\n    # Table variants, added a column for max minor allele frequency among gnomAD and TOPMed databases\n    c_vrt_unnested = ['max_gnomad_topmed']\n    c_vrt_nested = [\"topmed\", 'gnomad_genomes_2_1', 'gnomad_exomes_2_1', 'gnomad_genomes_3_0',\n                    'gnomad_genomes_3_1_1']\n    # c_vrt = ['max_gnomad_topmed', 'topmed', 'gnomad_genomes_2_1', 'gnomad_exomes_2_1', 'gnomad_genomes_3_0', 'gnomad_genomes_3_1_1']\n    t_vrt = variants \\\n        .withColumn('max_gnomad_topmed',\n                    F.greatest(F.lit(0), F.col('topmed')['af'], F.col('gnomad_exomes_2_1')['af'], \\\n                               F.col('gnomad_genomes_2_1')['af'], F.col('gnomad_genomes_3_0')['af'],\n                               F.col('gnomad_genomes_3_1_1')['af'])) \\\n        .where((F.size(F.array_intersect(F.col('studies'), F.lit(F.array(*map(F.lit, study_id_list))))) > 0) & \\\n               F.col('chromosome').isin(chr_list)) \\\n        .select(cond + c_vrt_unnested + [F.col(nc + '.' + c).alias(nc + '_' + c)\n                                         for nc in c_vrt_nested\n                                         for c in variants.select(nc + '.*').columns])\n\n    # Table ClinVar, restricted to those seen in variants and labeled as pathogenic/likely_pathogenic\n    c_clv = ['VariationID', 'clin_sig']\n    t_clv = clinvar \\\n        .withColumnRenamed('name', 'VariationID') \\\n        .where(F.split(F.split(F.col('geneinfo'), '\\\\|')[0], ':')[0].isin(gene_symbols_trunc) \\\n               & (F.array_contains(F.col('clin_sig'), 'Pathogenic') \\\n                  | F.array_contains(F.col('clin_sig'), 'Likely_pathogenic'))) \\\n        .join(t_vrt, cond) \\\n        .select(cond + c_clv)\n\n    # Table ClinVar, restricted to those seen in variants and labeled as pathogenic/likely_pathogenic/vus\n    # c_clv = ['VariationID', 'clin_sig']\n    # t_clv = spark \\\n    #     .table('clinvar') \\\n    #     .withColumnRenamed('name', 'VariationID') \\\n    #     .where(F.split(F.split(F.col('geneinfo'), '\\\\|')[0], ':')[0].isin(gene_symbols_trunc) \\\n    #         & (F.array_contains(F.col('clin_sig'), 'Pathogenic') \\\n    #         | F.array_contains(F.col('clin_sig'), 'Likely_pathogenic') \\\n    #         | F.array_contains(F.col('clin_sig'), 'Uncertain_significance'))) \\\n    #     .join(t_vrt, cond) \\\n    #     .select(cond + c_clv)\n\n    # Table HGMD, restricted to those seen in variants and labeled as DM or DM?\n    c_hgmd = ['HGMDID', 'variant_class']\n    t_hgmd = hg38_HGMD2022Q4_variant \\\n        .withColumnRenamed('id', 'HGMDID') \\\n        .where(F.col('symbol').isin(gene_symbols_trunc) \\\n               & F.col('variant_class').startswith('DM')) \\\n        .join(t_vrt, cond) \\\n        .select(cond + c_hgmd)\n\n    # Join consequences, variants and dbnsfp, restricted to those with MAF less than a threshold and PredCountRatio_D2T within a range\n    t_csq_vrt = t_csq \\\n        .join(t_vrt, cond)\n\n    t_csq_vrt_dbn = t_csq_vrt \\\n        .join(t_dbn, cond, how='left') \\\n        .withColumn('flag', F.when( \\\n        (F.col('max_gnomad_topmed') < gnomAD_TOPMed_maf) \\\n        & (F.col('PredCountRatio_D2T').isNull() \\\n           | (F.col('PredCountRatio_D2T').isNotNull() \\\n              & (F.col('PredCountRatio_D2T') >= dpc_l) \\\n              & (F.col('PredCountRatio_D2T') <= dpc_u)) \\\n           ), 1) \\\n                    .otherwise(0))\n\n    # Include ClinVar if specified\n    if 'ClinVar' in known_variants_l and t_clv.count() > 0:\n        t_csq_vrt_dbn = t_csq_vrt_dbn \\\n            .join(t_clv, cond, how='left') \\\n            .withColumn('flag', F.when(F.col('VariationID').isNotNull(), 1).otherwise(t_csq_vrt_dbn.flag))\n\n    # Include HGMD if specified\n    if 'HGMD' in known_variants_l and t_hgmd.count() > 0:\n        t_csq_vrt_dbn = t_csq_vrt_dbn \\\n            .join(t_hgmd, cond, how='left') \\\n            .withColumn('flag', F.when(F.col('HGMDID').isNotNull(), 1).otherwise(t_csq_vrt_dbn.flag))\n\n    # Table occurrences, restricted to input genes, chromosomes of those genes, input study IDs, and occurrences where alternate allele\n    # is present, plus adjusted calls based on alternative allele fraction in the total sequencing depth\n    c_ocr = ['ad', 'dp', 'calls', 'filter', 'is_lo_conf_denovo', 'is_hi_conf_denovo', 'is_proband',\n             'affected_status', 'gender', \\\n             'biospecimen_id', 'participant_id', 'mother_id', 'father_id', 'family_id']\n    t_ocr = occurrences \\\n        .withColumn('adjusted_calls',\n                    F.when(F.col('ad')[1] / (F.col('ad')[0] + F.col('ad')[1]) < aaf, F.array(F.lit(0), F.lit(0))) \\\n                    .otherwise(F.col('calls'))) \\\n        .where(F.col('chromosome').isin(chr_list) \\\n               & (F.col('is_multi_allelic') == 'false') \\\n               & (F.col('has_alt') == 1) \\\n               & (F.col('adjusted_calls') != F.array(F.lit(0), F.lit(0)))) \\\n        .select(cond + c_ocr)\n\n    # Finally join all together\n    # t_output = F.broadcast(t_csq_vrt_dbn) \\\n    t_output = t_csq_vrt_dbn \\\n        .join(t_ocr, cond) \\\n        .where(t_csq_vrt_dbn.flag == 1) \\\n        .drop('flag')\n\n    t_dgn = diagnoses \\\n        .select('participant_id', 'source_text_diagnosis') \\\n        .distinct() \\\n        .groupBy('participant_id') \\\n        .agg(F.collect_list('source_text_diagnosis').alias('diagnoses_combined'))\n    t_pht = phenotypes \\\n        .select('participant_id', 'source_text_phenotype', 'hpo_id_phenotype') \\\n        .distinct() \\\n        .groupBy('participant_id') \\\n        .agg(F.collect_list('source_text_phenotype').alias('phenotypes_combined'), \\\n             F.collect_list('hpo_id_phenotype').alias('hpos_combined'))\n    t_output_dgn_pht = t_output \\\n        .join(t_dgn, 'participant_id', 'left') \\\n        .join(t_pht, 'participant_id', 'left')\n\n    return (t_output_dgn_pht)\n\n\n# define output name and write table t_output\ndef write_output(t_output, args):\n    # parameter configuration\n    gene_symbols_trunc = args.gene_symbols\n    study_id_list = args.study_ids\n\n    studies = spark.read.parquet(args.studies)\n    short_code_id = list(studies \\\n                         .where(F.col('kf_id').isin(study_id_list)) \\\n                         .select('short_code').toPandas()['short_code'])\n    Date = list(spark.sql(\"select current_date()\") \\\n                .withColumn(\"current_date()\", F.col(\"current_date()\").cast(\"string\")) \\\n                .toPandas()['current_date()'])\n    t_output.toPandas() \\\n        .to_csv(\"/sbgenomics/output-files/\" + \"_\".join(Date + short_code_id + gene_symbols_trunc) +\".tsv\", sep=\"\\t\", index=False, na_rep='-')\n    # print(\"/sbgenomics/output-files/\" + \"_\".join(Date + short_code_id + gene_symbols_trunc) +\".tsv\")\n\n\nt_output = gene_based_filt(args)\nwrite_output(t_output, args)",
                    "writable": false
                }
            ]
        },
        {
            "class": "InlineJavascriptRequirement"
        }
    ],
    "sbg:projectName": "Variant WorkBench testing",
    "sbg:revisionsInfo": [
        {
            "sbg:revision": 0,
            "sbg:modifiedBy": "qqlii44",
            "sbg:modifiedOn": 1678475329,
            "sbg:revisionNotes": null
        },
        {
            "sbg:revision": 1,
            "sbg:modifiedBy": "qqlii44",
            "sbg:modifiedOn": 1678903021,
            "sbg:revisionNotes": ""
        },
        {
            "sbg:revision": 2,
            "sbg:modifiedBy": "qqlii44",
            "sbg:modifiedOn": 1678903621,
            "sbg:revisionNotes": ""
        },
        {
            "sbg:revision": 3,
            "sbg:modifiedBy": "qqlii44",
            "sbg:modifiedOn": 1678904037,
            "sbg:revisionNotes": ""
        },
        {
            "sbg:revision": 4,
            "sbg:modifiedBy": "qqlii44",
            "sbg:modifiedOn": 1678909599,
            "sbg:revisionNotes": ""
        },
        {
            "sbg:revision": 5,
            "sbg:modifiedBy": "qqlii44",
            "sbg:modifiedOn": 1678911067,
            "sbg:revisionNotes": ""
        },
        {
            "sbg:revision": 6,
            "sbg:modifiedBy": "qqlii44",
            "sbg:modifiedOn": 1678911283,
            "sbg:revisionNotes": ""
        },
        {
            "sbg:revision": 7,
            "sbg:modifiedBy": "qqlii44",
            "sbg:modifiedOn": 1678913724,
            "sbg:revisionNotes": "Python"
        },
        {
            "sbg:revision": 8,
            "sbg:modifiedBy": "brownm28",
            "sbg:modifiedOn": 1680283440,
            "sbg:revisionNotes": "added docker image"
        },
        {
            "sbg:revision": 9,
            "sbg:modifiedBy": "qqlii44",
            "sbg:modifiedOn": 1680283953,
            "sbg:revisionNotes": "added docker image"
        },
        {
            "sbg:revision": 10,
            "sbg:modifiedBy": "qqlii44",
            "sbg:modifiedOn": 1680290089,
            "sbg:revisionNotes": "debug pd.set_option('max_columns', None) pandas._config.config.OptionError: 'Pattern matched multiple keys'"
        },
        {
            "sbg:revision": 11,
            "sbg:modifiedBy": "qqlii44",
            "sbg:modifiedOn": 1680294186,
            "sbg:revisionNotes": "used spark-submit"
        },
        {
            "sbg:revision": 12,
            "sbg:modifiedBy": "qqlii44",
            "sbg:modifiedOn": 1680295624,
            "sbg:revisionNotes": "tried to solve \"module not found: io.projectglow#glow-spark3_2.12;1.1.2"
        },
        {
            "sbg:revision": 13,
            "sbg:modifiedBy": "qqlii44",
            "sbg:modifiedOn": 1680555415,
            "sbg:revisionNotes": "new image built from dockfile"
        },
        {
            "sbg:revision": 14,
            "sbg:modifiedBy": "qqlii44",
            "sbg:modifiedOn": 1680710140,
            "sbg:revisionNotes": "removed hard-coded paths/variables"
        },
        {
            "sbg:revision": 15,
            "sbg:modifiedBy": "qqlii44",
            "sbg:modifiedOn": 1680710782,
            "sbg:revisionNotes": "removed duplicated rows"
        },
        {
            "sbg:revision": 16,
            "sbg:modifiedBy": "qqlii44",
            "sbg:modifiedOn": 1680712780,
            "sbg:revisionNotes": "used miguel's way to create spark session"
        },
        {
            "sbg:revision": 17,
            "sbg:modifiedBy": "qqlii44",
            "sbg:modifiedOn": 1680714061,
            "sbg:revisionNotes": "moved sql.timeout config to base command"
        },
        {
            "sbg:revision": 18,
            "sbg:modifiedBy": "qqlii44",
            "sbg:modifiedOn": 1680722263,
            "sbg:revisionNotes": "changed to miguel's docker image"
        },
        {
            "sbg:revision": 19,
            "sbg:modifiedBy": "qqlii44",
            "sbg:modifiedOn": 1680723016,
            "sbg:revisionNotes": "remove spark.sql.broadcastTimeout"
        },
        {
            "sbg:revision": 20,
            "sbg:modifiedBy": "brownm28",
            "sbg:modifiedOn": 1680724442,
            "sbg:revisionNotes": "added a default input"
        }
    ],
    "sbg:image_url": null,
    "sbg:toolAuthor": "Qi Li",
    "sbg:appVersion": [
        "v1.2"
    ],
    "sbg:id": "yiran/variant-workbench-testing/gene-based-variant-filtering/20",
    "sbg:revision": 20,
    "sbg:revisionNotes": "added a default input",
    "sbg:modifiedOn": 1680724442,
    "sbg:modifiedBy": "brownm28",
    "sbg:createdOn": 1678475329,
    "sbg:createdBy": "qqlii44",
    "sbg:project": "yiran/variant-workbench-testing",
    "sbg:sbgMaintained": false,
    "sbg:validationErrors": [],
    "sbg:contributors": [
        "qqlii44",
        "brownm28"
    ],
    "sbg:latestRevision": 20,
    "sbg:publisher": "sbg",
    "sbg:content_hash": "ab1a0ca699ffeeeacb3e6e80a2456b64b216749e7b4a202329b530c30ad67611f",
    "sbg:workflowLanguage": "CWL"
}