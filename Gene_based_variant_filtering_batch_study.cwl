{
    "class": "CommandLineTool",
    "cwlVersion": "v1.2",
    "$namespaces": {
        "sbg": "https://sevenbridges.com"
    },
    "baseCommand": [
        "spark-submit"
    ],
    "inputs": [
        {
            "id": "gene_list",
            "type": "File",
            "inputBinding": {
                "prefix": "-g",
                "shellQuote": false,
                "position": 1
            },
            "doc": "A text file that contains the list of gene. Each row in the text file should correspond to one gene. No header required."
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
                "path": "64b552738d68fe19595831bd",
                "name": "hg38_HGMD2023Q2_variant"
            },
            "id": "hgmd_parquet",
            "type": "Directory",
            "inputBinding": {
                "prefix": "--hgmd",
                "shellQuote": false,
                "position": 0
            },
            "doc": "the latest HGMD variant  parquet file dir"
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
            "sbg:suggestedValue": {
                "class": "File",
                "path": "647e3be91dfc710d249ff2d4",
                "name": "part-00000-cb3a24a6-5009-48fd-9fc0-a33c2034b376-c000.snappy.parquet"
            },
            "id": "clinvar_parquet",
            "type": "File",
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
            "sbg:suggestedValue": {
                "class": "File",
                "path": "63ff628bf897792af067fa30",
                "name": "part-00000-6acc32c0-8485-44e0-978e-bc18e9352900-c000.snappy.parquet"
            },
            "id": "diagnoses_parquet",
            "type": "File",
            "inputBinding": {
                "prefix": "--diagnoses",
                "shellQuote": false,
                "position": 0
            }
        },
        {
            "sbg:suggestedValue": {
                "class": "File",
                "path": "63ff628ef897792af067ffe7",
                "name": "part-00000-52cc98f2-8ecd-403b-a153-c08b63be5d8a-c000.snappy.parquet"
            },
            "id": "phenotypes_parquet",
            "type": "File",
            "inputBinding": {
                "prefix": "--phenotypes",
                "shellQuote": false,
                "position": 0
            },
            "doc": "phenotypes parquet file dir"
        },
        {
            "id": "gnomAD_TOPMed_maf",
            "type": "double?",
            "inputBinding": {
                "prefix": "--maf",
                "shellQuote": false,
                "position": 0
            },
            "doc": "the max global AF across all gnomAD and TOPMed databases",
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
            "id": "alternative_allele_fraction",
            "type": "double?",
            "inputBinding": {
                "prefix": "--aaf",
                "shellQuote": false,
                "position": 0
            },
            "doc": "alternative allele fraction",
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
            "id": "sql_broadcastTimeout",
            "type": "int?",
            "doc": ".config(\"spark.sql.broadcastTimeout\", 36000)",
            "default": 36000
        },
        {
            "loadListing": "deep_listing",
            "id": "occurrences",
            "type": "Directory",
            "inputBinding": {
                "prefix": "--occurrences",
                "shellQuote": false,
                "position": 0
            },
            "doc": "the parent directory of occurrences parquet directory"
        },
        {
            "id": "output_basename",
            "type": "string?",
            "inputBinding": {
                "prefix": "--output_basename",
                "shellQuote": false,
                "position": 0
            },
            "doc": "Recommand use the task ID in the url above as output file prefix. \nFor example 598b5c92-cb1d-49b2-8030-e1aa3e9b9fde is the task ID from \n\t    https://cavatica.sbgenomics.com/u/yiran/variant-workbench-testing/tasks/598b5c92-cb1d-49b2-8030-e1aa3e9b9fde/#set-input-data",
            "default": "gene-based-variant-filtering"
        }
    ],
    "outputs": [
        {
            "id": "variants_output",
            "type": "File?",
            "outputBinding": {
                "glob": "*.tsv.gz"
            }
        }
    ],
    "doc": "Get a list of deleterious variants in interested genes from specified study cohort(s) in the Kids First program.",
    "label": "Gene_Based_Variant_Filtering",
    "arguments": [
        {
            "prefix": "",
            "shellQuote": false,
            "position": 0,
            "valueFrom": "--packages io.projectglow:glow-spark3_2.12:1.1.2  --conf spark.hadoop.io.compression.codecs=io.projectglow.sql.util.BGZFCodec --conf spark.sql.broadcastTimeout=$(inputs.sql_broadcastTimeout) --driver-memory $(inputs.spark_driver_mem)G  Gene_based_variant_filtering_batch_study.py"
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
            "class": "ResourceRequirement",
            "coresMin": 16
        },
        {
            "class": "DockerRequirement",
            "dockerPull": "pgc-images.sbgenomics.com/d3b-bixu/pyspark:3.1.2"
        },
        {
            "class": "InitialWorkDirRequirement",
            "listing": [
                {
                    "entryname": "Gene_based_variant_filtering_batch_study.py",
                    "entry": "import argparse\nfrom argparse import RawTextHelpFormatter\nimport pyspark\nfrom pyspark.sql import functions as F\nfrom pyspark.sql.types import DoubleType\nimport glow\nimport sys\n\nparser = argparse.ArgumentParser(\n    description='Script of gene based variant filtering. \\n\\\n    MUST BE RUN WITH spark-submit. For example: \\n\\\n    spark-submit --driver-memory 10G Gene_based_variant_filtering.py',\n    formatter_class=RawTextHelpFormatter)\n\nparser = argparse.ArgumentParser()\nparser.add_argument('-g', '--gene_list_file', required=True,\n                    help='A text file that contains the list of gene. Each row in the text file should correspond to one gene. No header required.')\n# parser.add_argument('-s', '--study_ids', nargs='+', default=[], required=True,\n#                     help='list of study to look for')\nparser.add_argument('--hgmd',\n        help='HGMD variant parquet file dir')\nparser.add_argument('--dbnsfp',\n        help='dbnsfp annovar parquet file dir')\nparser.add_argument('--clinvar',\n        help='clinvar parquet file dir')\nparser.add_argument('--consequences',\n        help='consequences parquet file dir')\nparser.add_argument('--variants',\n        help='variants parquet file dir')\nparser.add_argument('--diagnoses',\n        help='diagnoses parquet file dir')\nparser.add_argument('--phenotypes',\n        help='phenotypes parquet file dir')\nparser.add_argument('--occurrences',\n        help='occurrences parquet file dir')\nparser.add_argument('--maf', default=0.0001,\n        help='gnomAD and TOPMed max allele frequency')\nparser.add_argument('--dpc_l', default=0.5,\n        help='damage predict count lower threshold')\nparser.add_argument('--dpc_u', default=1,\n        help='damage predict count upper threshold')\nparser.add_argument('--known_variants_l', nargs='+', default=['ClinVar', 'HGMD'],\n                    help='known variant databases used, default is ClinVar and HGMD')\nparser.add_argument('--aaf', default=0.2,\n        help='alternative allele fraction threshold')\nparser.add_argument('--output_basename', default='gene-based-variant-filtering',\n        help='Recommand use the task ID in the url above as output file prefix. \\\n        For example 598b5c92-cb1d-49b2-8030-e1aa3e9b9fde is the task ID from \\\n\t    https://cavatica.sbgenomics.com/u/yiran/variant-workbench-testing/tasks/598b5c92-cb1d-49b2-8030-e1aa3e9b9fde/#set-input-data')\n\nargs = parser.parse_args()\n\n# Create spark session\nspark = (\n    pyspark.sql.SparkSession.builder.appName(\"PythonPi\")\n    .getOrCreate()\n    )\n# Register so that glow functions like read vcf work with spark. Must be run in spark shell or in context described in help\nspark = glow.register(spark)\n\n# parameter configuration\ngene_text_path = args.gene_list_file\n# study_id_list = args.study_ids\ngnomAD_TOPMed_maf = args.maf\ndpc_l = args.dpc_l\ndpc_u = args.dpc_u\nknown_variants_l = args.known_variants_l\naaf = args.aaf\noutput_basename = args.output_basename\noccurrences_path = args.occurrences\n\n# get a list of interested gene and remove unwanted strings in the end of each gene\ngene_symbols_trunc = spark.read.option(\"header\", False).text(gene_text_path)\ngene_symbols_trunc = list(gene_symbols_trunc.toPandas()['value'])\ngene_symbols_trunc = [gl.replace('\\xa0', '').replace('\\n', '') for gl in gene_symbols_trunc]\n\n# customized tables loading\nhg38_HGMD_variant = spark.read.parquet(args.hgmd)\ndbnsfp_annovar = spark.read.parquet(args.dbnsfp)\nclinvar = spark.read.parquet(args.clinvar)\nconsequences = spark.read.parquet(args.consequences)\nvariants = spark.read.parquet(args.variants)\ndiagnoses = spark.read.parquet(args.diagnoses)\nphenotypes = spark.read.parquet(args.phenotypes)\noccurrences = spark.read.parquet(occurrences_path)\n\n# provided study id by occurences table\n# Find the start and end index of the desired substring\nstart_index = occurrences_path.index(\"occurrences_sd\") + len(\"occurrences_sd\")\nend_index = occurrences_path.index(\"_re_\", start_index)\n# Extract the desired substring\nsubstring = occurrences_path[start_index-2:end_index]\n# Convert the substring to uppercase\nstudy_id_list = [substring.upper()]\n\n# read multi studies\n# occ_dict = []\n# for s_id in study_id_list:\n#     occ_dict.append(spark.read.parquet(occurrences_parent_path + '/occurrences_*/study_id=' + s_id))\n# occurrences = reduce(DataFrame.unionAll, occ_dict)\n\n# gene based variant filtering\ndef gene_based_filt(gene_symbols_trunc, study_id_list, gnomAD_TOPMed_maf, dpc_l, dpc_u,\n\t\t            known_variants_l, aaf, hg38_HGMD_variant, dbnsfp_annovar, clinvar, \n\t                consequences, variants, diagnoses, phenotypes, occurrences):\n    #  Actual running step, generating table t_output\n    cond = ['chromosome', 'start', 'reference', 'alternate']\n\n    # Table consequences, restricted to canonical annotation and input genes/study IDs\n    c_csq = ['consequences', 'rsID', 'impact', 'symbol', 'ensembl_gene_id', 'refseq_mrna_id', 'hgvsg', 'hgvsc',\n             'hgvsp', 'study_ids']\n    t_csq = consequences.withColumnRenamed('name', 'rsID') \\\n        .drop('variant_class') \\\n        .where((F.col('original_canonical') == 'true') \\\n               & (F.col('symbol').isin(gene_symbols_trunc)) \\\n               & (F.size(F.array_intersect(F.col('study_ids'), F.lit(F.array(*map(F.lit, study_id_list))))) > 0)) \\\n        .select(cond + c_csq)\n    chr_list = [c['chromosome'] for c in t_csq.select('chromosome').distinct().collect()]\n\n    # Table dbnsfp_annovar, added a column for ratio of damage predictions to all predictions\n    c_dbn = ['DamagePredCount', 'PredCountRatio_D2T', 'TWINSUK_AF', 'ALSPAC_AF', 'UK10K_AF']\n    t_dbn = dbnsfp_annovar \\\n        .where(F.col('chromosome').isin(chr_list)) \\\n        .withColumn('PredCountRatio_D2T',\n                    F.when(F.split(F.col('DamagePredCount'), '_')[1] == 0, F.lit(None).cast(DoubleType())) \\\n                    .otherwise(\n                        F.split(F.col('DamagePredCount'), '_')[0] / F.split(F.col('DamagePredCount'), '_')[1])) \\\n        .select(cond + c_dbn)\n\n    # Table variants, added a column for max minor allele frequency among gnomAD and TOPMed databases\n    c_vrt_unnested = ['max_gnomad_topmed']\n    c_vrt_nested = [\"topmed\", 'gnomad_genomes_2_1', 'gnomad_exomes_2_1', 'gnomad_genomes_3_0',\n                    'gnomad_genomes_3_1_1']\n    # c_vrt = ['max_gnomad_topmed', 'topmed', 'gnomad_genomes_2_1', 'gnomad_exomes_2_1', 'gnomad_genomes_3_0', 'gnomad_genomes_3_1_1']\n    t_vrt = variants \\\n        .withColumn('max_gnomad_topmed',\n                    F.greatest(F.lit(0), F.col('topmed')['af'], F.col('gnomad_exomes_2_1')['af'], \\\n                               F.col('gnomad_genomes_2_1')['af'], F.col('gnomad_genomes_3_0')['af'],\n                               F.col('gnomad_genomes_3_1_1')['af'])) \\\n        .where((F.size(F.array_intersect(F.col('studies'), F.lit(F.array(*map(F.lit, study_id_list))))) > 0) & \\\n               F.col('chromosome').isin(chr_list)) \\\n        .select(*[F.expr(f\"{nc}.af\").alias(f\"{nc}_af\") for nc in c_vrt_nested] + cond + c_vrt_unnested)\n\n    # Table ClinVar, restricted to those seen in variants and labeled as pathogenic/likely_pathogenic\n    c_clv = ['VariationID', 'clin_sig', 'conditions']\n    t_clv = clinvar \\\n        .withColumnRenamed('name', 'VariationID') \\\n        .where(F.split(F.split(F.col('geneinfo'), '\\\\|')[0], ':')[0].isin(gene_symbols_trunc) \\\n               & (F.array_contains(F.col('clin_sig'), 'Pathogenic') \\\n                  | F.array_contains(F.col('clin_sig'), 'Likely_pathogenic'))) \\\n        .join(t_vrt, cond) \\\n        .select(cond + c_clv)\n\n    # Table ClinVar, restricted to those seen in variants and labeled as pathogenic/likely_pathogenic/vus\n    # c_clv = ['VariationID', 'clin_sig']\n    # t_clv = spark \\\n    #     .table('clinvar') \\\n    #     .withColumnRenamed('name', 'VariationID') \\\n    #     .where(F.split(F.split(F.col('geneinfo'), '\\\\|')[0], ':')[0].isin(gene_symbols_trunc) \\\n    #         & (F.array_contains(F.col('clin_sig'), 'Pathogenic') \\\n    #         | F.array_contains(F.col('clin_sig'), 'Likely_pathogenic') \\\n    #         | F.array_contains(F.col('clin_sig'), 'Uncertain_significance'))) \\\n    #     .join(t_vrt, cond) \\\n    #     .select(cond + c_clv)\n\n    # Table HGMD, restricted to those seen in variants and labeled as DM or DM?\n    c_hgmd = ['HGMDID', 'variant_class', 'phen']\n    t_hgmd = hg38_HGMD_variant \\\n        .withColumnRenamed('id', 'HGMDID') \\\n        .where(F.col('symbol').isin(gene_symbols_trunc) \\\n               & F.col('variant_class').startswith('DM')) \\\n        .join(t_vrt, cond) \\\n        .select(cond + c_hgmd)\n\n    # Join consequences, variants and dbnsfp, restricted to those with MAF less than a threshold and PredCountRatio_D2T within a range\n    t_csq_vrt = t_csq \\\n        .join(t_vrt, cond)\n\n    t_csq_vrt_dbn = t_csq_vrt \\\n        .join(t_dbn, cond, how='left') \\\n        .withColumn('flag', F.when( \\\n        (F.col('max_gnomad_topmed') < gnomAD_TOPMed_maf) \\\n        & (F.col('PredCountRatio_D2T').isNull() \\\n           | (F.col('PredCountRatio_D2T').isNotNull() \\\n              & (F.col('PredCountRatio_D2T') >= dpc_l) \\\n              & (F.col('PredCountRatio_D2T') <= dpc_u)) \\\n           ), 1) \\\n                    .otherwise(0))\n\n    # Include ClinVar if specified\n    if 'ClinVar' in known_variants_l and t_clv.count() > 0:\n        t_csq_vrt_dbn = t_csq_vrt_dbn \\\n            .join(t_clv, cond, how='left') \\\n            .withColumn('flag', F.when(F.col('VariationID').isNotNull(), 1).otherwise(t_csq_vrt_dbn.flag))\n\n    # Include HGMD if specified\n    if 'HGMD' in known_variants_l and t_hgmd.count() > 0:\n        t_csq_vrt_dbn = t_csq_vrt_dbn \\\n            .join(t_hgmd, cond, how='left') \\\n            .withColumn('flag', F.when(F.col('HGMDID').isNotNull(), 1).otherwise(t_csq_vrt_dbn.flag))\n\n    # Table occurrences, restricted to input genes, chromosomes of those genes, input study IDs, and occurrences where alternate allele\n    # is present, plus adjusted calls based on alternative allele fraction in the total sequencing depth\n    c_ocr = ['ad', 'dp', 'variant_allele_fraction', 'calls', 'adjusted_calls', 'filter', 'is_lo_conf_denovo', 'is_hi_conf_denovo',\n\t     'is_proband', 'affected_status', 'gender',\n\t     'biospecimen_id', 'participant_id', 'mother_id', 'father_id', 'family_id']\n    t_ocr = occurrences.withColumn('variant_allele_fraction', F.col('ad')[1] / (F.col('ad')[0] + F.col('ad')[1])) \\\n        .withColumn('adjusted_calls', F.when(F.col('variant_allele_fraction') < aaf, F.array(F.lit(0), F.lit(0)))\n                                      .otherwise(F.col('calls'))) \\\n        .where(F.col('chromosome').isin(chr_list) \\\n               & (F.col('is_multi_allelic') == 'false') \\\n               & (F.col('has_alt') == 1) \\\n               & (F.col('adjusted_calls') != F.array(F.lit(0), F.lit(0)))) \\\n        .select(cond + c_ocr)\n\n    # Finally join all together\n    # t_output = F.broadcast(t_csq_vrt_dbn) \\\n    t_output = t_csq_vrt_dbn \\\n        .join(t_ocr, cond) \\\n        .where(t_csq_vrt_dbn.flag == 1) \\\n        .drop('flag')\n\n    t_dgn = diagnoses \\\n        .select('participant_id', 'source_text_diagnosis') \\\n        .distinct() \\\n        .groupBy('participant_id') \\\n        .agg(F.collect_list('source_text_diagnosis').alias('diagnoses_combined'))\n    t_pht = phenotypes \\\n        .select('participant_id', 'source_text_phenotype', 'hpo_id_phenotype') \\\n        .distinct() \\\n        .groupBy('participant_id') \\\n        .agg(F.collect_list('source_text_phenotype').alias('phenotypes_combined'), \\\n             F.collect_list('hpo_id_phenotype').alias('hpos_combined'))\n    t_output_dgn_pht = t_output \\\n        .join(t_dgn, 'participant_id', 'left') \\\n        .join(t_pht, 'participant_id', 'left')\n\n    return (t_output_dgn_pht)\n\n\n# define output name and write table t_output\ndef write_output(t_output, output_basename, study_id_list):\n    Date = list(spark.sql(\"select current_date()\") \\\n                .withColumn(\"current_date()\", F.col(\"current_date()\").cast(\"string\")) \\\n                .toPandas()['current_date()'])\n    output_filename= \"_\".join(Date) + \"_\" + output_basename + \"_\" + study_id_list + \".tsv.gz\"\n    t_output.toPandas() \\\n        .to_csv(output_filename, sep=\"\\t\", index=False, na_rep='-', compression='gzip')\n\nif args.hgmd is None:\n\tprint(\"Missing hgmd parquet file\", file=sys.stderr)\n\texit(1)\nif args.dbnsfp is None:\n\tprint(\"Missing dbnsfp parquet file\", file=sys.stderr)\n\texit(1)\nif args.clinvar is None:\n\tprint(\"Missing clinvar parquet file\", file=sys.stderr)\n\texit(1)\nif args.consequences is None:\n\tprint(\"Missing consequences parquet file\", file=sys.stderr)\n\texit(1)\nif args.variants is None:\n\tprint(\"Missing variants parquet file\", file=sys.stderr)\n\texit(1)\nif args.diagnoses is None:\n\tprint(\"Missing diagnoses parquet file\", file=sys.stderr)\n\texit(1)\nif args.phenotypes is None:\n\tprint(\"Missing phenotypes parquet file\", file=sys.stderr)\n\texit(1)\n\t\nt_output = gene_based_filt(gene_symbols_trunc, study_id_list, gnomAD_TOPMed_maf, dpc_l, dpc_u,\n\t\t                known_variants_l, aaf, hg38_HGMD_variant, dbnsfp_annovar, clinvar, \n\t                    consequences, variants, diagnoses, phenotypes, occurrences)\nwrite_output(t_output, output_basename, study_id_list)",
                    "writable": true
                }
            ]
        },
        {
            "class": "InlineJavascriptRequirement"
        }
    ],
    "hints": [
        {
            "class": "sbg:maxNumberOfParallelInstances",
            "value": "6"
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
        },
        {
            "sbg:revision": 21,
            "sbg:modifiedBy": "qqlii44",
            "sbg:modifiedOn": 1680733045,
            "sbg:revisionNotes": "added all default input"
        },
        {
            "sbg:revision": 22,
            "sbg:modifiedBy": "qqlii44",
            "sbg:modifiedOn": 1680733224,
            "sbg:revisionNotes": "turned off maf required option"
        },
        {
            "sbg:revision": 23,
            "sbg:modifiedBy": "qqlii44",
            "sbg:modifiedOn": 1680737636,
            "sbg:revisionNotes": "added occurrences input"
        },
        {
            "sbg:revision": 24,
            "sbg:modifiedBy": "qqlii44",
            "sbg:modifiedOn": 1681160054,
            "sbg:revisionNotes": "use task id as ouput basename"
        },
        {
            "sbg:revision": 25,
            "sbg:modifiedBy": "qqlii44",
            "sbg:modifiedOn": 1681224678,
            "sbg:revisionNotes": "add min CPU, genes file input, changed output name, multi studies"
        },
        {
            "sbg:revision": 26,
            "sbg:modifiedBy": "qqlii44",
            "sbg:modifiedOn": 1681313418,
            "sbg:revisionNotes": "increased maxNumberOfParallelInstances"
        },
        {
            "sbg:revision": 27,
            "sbg:modifiedBy": "yiran",
            "sbg:modifiedOn": 1682525622,
            "sbg:revisionNotes": "Updated App Description"
        },
        {
            "sbg:revision": 28,
            "sbg:modifiedBy": "qqlii44",
            "sbg:modifiedOn": 1683124305,
            "sbg:revisionNotes": "added/removed columns"
        },
        {
            "sbg:revision": 29,
            "sbg:modifiedBy": "qqlii44",
            "sbg:modifiedOn": 1683124648,
            "sbg:revisionNotes": "updated latest Clinvar and HGMD"
        },
        {
            "sbg:revision": 30,
            "sbg:modifiedBy": "qqlii44",
            "sbg:modifiedOn": 1687550711,
            "sbg:revisionNotes": "change parquet dirs to files"
        },
        {
            "sbg:revision": 31,
            "sbg:modifiedBy": "qqlii44",
            "sbg:modifiedOn": 1687551293,
            "sbg:revisionNotes": "single study only"
        },
        {
            "sbg:revision": 32,
            "sbg:modifiedBy": "qqlii44",
            "sbg:modifiedOn": 1687551498,
            "sbg:revisionNotes": "single study only"
        },
        {
            "sbg:revision": 33,
            "sbg:modifiedBy": "qqlii44",
            "sbg:modifiedOn": 1688591280,
            "sbg:revisionNotes": "debugged"
        },
        {
            "sbg:revision": 34,
            "sbg:modifiedBy": "qqlii44",
            "sbg:modifiedOn": 1688654775,
            "sbg:revisionNotes": "added column 'variant_allele_fraction'"
        },
        {
            "sbg:revision": 35,
            "sbg:modifiedBy": "qqlii44",
            "sbg:modifiedOn": 1689368529,
            "sbg:revisionNotes": "batch_task-study"
        },
        {
            "sbg:revision": 36,
            "sbg:modifiedBy": "qqlii44",
            "sbg:modifiedOn": 1689606436,
            "sbg:revisionNotes": "debug and hgmd2023Q2"
        },
        {
            "sbg:revision": 37,
            "sbg:modifiedBy": "qqlii44",
            "sbg:modifiedOn": 1689609037,
            "sbg:revisionNotes": "debug"
        }
    ],
    "sbg:image_url": null,
    "sbg:toolAuthor": "Qi Li",
    "sbg:appVersion": [
        "v1.2"
    ],
    "id": "https://cavatica-api.sbgenomics.com/v2/apps/yiran/variant-workbench-testing/gene-based-variant-filtering/37/raw/",
    "sbg:id": "yiran/variant-workbench-testing/gene-based-variant-filtering/37",
    "sbg:revision": 37,
    "sbg:revisionNotes": "debug",
    "sbg:modifiedOn": 1689609037,
    "sbg:modifiedBy": "qqlii44",
    "sbg:createdOn": 1678475329,
    "sbg:createdBy": "qqlii44",
    "sbg:project": "yiran/variant-workbench-testing",
    "sbg:sbgMaintained": false,
    "sbg:validationErrors": [],
    "sbg:contributors": [
        "yiran",
        "brownm28",
        "qqlii44"
    ],
    "sbg:latestRevision": 37,
    "sbg:publisher": "sbg",
    "sbg:content_hash": "aa254fa60de7e8c51749bd131db68b7f32895627ca4386a96102be7e9dcef4276",
    "sbg:workflowLanguage": "CWL"
}