import argparse
from argparse import RawTextHelpFormatter
import pyspark
from pyspark.sql import DataFrame, functions as F
from pyspark.sql.types import DoubleType
import glow
import sys
# from functools import reduce

parser = argparse.ArgumentParser(
    description='Script of gene based variant filtering. \n\
    MUST BE RUN WITH spark-submit. For example: \n\
    spark-submit --driver-memory 10G Gene_based_variant_filtering.py',
    formatter_class=RawTextHelpFormatter)

parser = argparse.ArgumentParser()
parser.add_argument('-g', '--gene_list_file',
                    help='A text file that contains the list of gene. Each row in the text file should correspond to one gene. No header required.')
parser.add_argument('-s', '--study_ids', nargs='+', default=[],
                    help='list of study to look for')
parser.add_argument('--hgmd',
        help='HGMD variant parquet file dir')
parser.add_argument('--dbnsfp',
        help='dbnsfp annovar parquet file dir')
parser.add_argument('--clinvar',
        help='clinvar parquet file dir')
parser.add_argument('--consequences',
        help='consequences parquet file dir')
parser.add_argument('--variants',
        help='variants parquet file dir')
parser.add_argument('--diagnoses',
        help='diagnoses parquet file dir')
parser.add_argument('--phenotypes',
        help='phenotypes parquet file dir')
parser.add_argument('--occurrences',
        help='occurrences parquet file dir')
parser.add_argument('--studies',
        help='studies parquet file dir')
parser.add_argument('--maf',
        help='gnomAD and TOPMed max allele frequency')
parser.add_argument('--dpc_l',
        help='damage predict count lower threshold')
parser.add_argument('--dpc_u',
        help='damage predict count upper threshold')
parser.add_argument('--known_variants_l', nargs='+', default=['ClinVar', 'HGMD'],
                    help='known variant databases used, default is ClinVar and HGMD')
parser.add_argument('--aaf',
        help='alternative allele frequency threshold')

args = parser.parse_args()

# Create spark session
spark = (
    pyspark.sql.SparkSession.builder.appName("PythonPi")
    .getOrCreate()
    )
# Register so that glow functions like read vcf work with spark. Must be run in spark shell or in context described in help
spark = glow.register(spark)

# parameter configuration
gene_text_path = args.gene_list_file
study_id_list = args.study_ids
gnomAD_TOPMed_maf = args.maf
dpc_l = args.dpc_l
dpc_u = args.dpc_u
known_variants_l = args.known_variants_l
aaf = args.aaf

# get a list of interested gene and remove unwanted strings in the end of each gene
gene_symbols_trunc = spark.read.option("header", False).text(gene_text_path)
gene_symbols_trunc = list(gene_symbols_trunc.toPandas()['value'])
gene_symbols_trunc = [gl.replace('\xa0', '').replace('\n', '') for gl in gene_symbols_trunc]

# customized tables loading
hg38_HGMD2022Q4_variant = spark.read.parquet(args.hgmd)
dbnsfp_annovar = spark.read.parquet(args.dbnsfp)
clinvar = spark.read.parquet(args.clinvar)
consequences = spark.read.parquet(args.consequences)
variants = spark.read.parquet(args.variants)
diagnoses = spark.read.parquet(args.diagnoses)
phenotypes = spark.read.parquet(args.phenotypes)
occurrences = spark.read.parquet(args.occurrences)
studies = spark.read.parquet(args.studies)

## read multi studies
# occ_dict = []
# for s_id in study_id_list:
#     occ_dict.append(spark.read.parquet('/sbgenomics/project-files/occurrences_Yiran/occurrences_*/study_id=' + s_id))
# occurrences = reduce(DataFrame.unionAll, occ_dict)

# gene based variant filtering
def gene_based_filt(gene_symbols_trunc, study_id_list, gnomAD_TOPMed_maf, dpc_l, dpc_u,
		            known_variants_l, aaf, hg38_HGMD2022Q4_variant, dbnsfp_annovar, clinvar, 
	                consequences, variants, diagnoses, phenotypes, occurrences):
    #  Actual running step, generating table t_output
    cond = ['chromosome', 'start', 'reference', 'alternate']

    # Table consequences, restricted to canonical annotation and input genes/study IDs
    c_csq = ['consequences', 'rsID', 'impact', 'symbol', 'ensembl_gene_id', 'refseq_mrna_id', 'hgvsg', 'hgvsc',
             'hgvsp', 'study_ids']
    t_csq = consequences.withColumnRenamed('name', 'rsID') \
        .drop('variant_class') \
        .where((F.col('original_canonical') == 'true') \
               & (F.col('symbol').isin(gene_symbols_trunc)) \
               & (F.size(F.array_intersect(F.col('study_ids'), F.lit(F.array(*map(F.lit, study_id_list))))) > 0)) \
        .select(cond + c_csq)
    chr_list = [c['chromosome'] for c in t_csq.select('chromosome').distinct().collect()]

    # Table dbnsfp_annovar, added a column for ratio of damage predictions to all predictions
    c_dbn = ['DamagePredCount', 'PredCountRatio_D2T', 'TWINSUK_AF', 'ALSPAC_AF', 'UK10K_AF']
    t_dbn = dbnsfp_annovar \
        .where(F.col('chromosome').isin(chr_list)) \
        .withColumn('PredCountRatio_D2T',
                    F.when(F.split(F.col('DamagePredCount'), '_')[1] == 0, F.lit(None).cast(DoubleType())) \
                    .otherwise(
                        F.split(F.col('DamagePredCount'), '_')[0] / F.split(F.col('DamagePredCount'), '_')[1])) \
        .select(cond + c_dbn)

    # Table variants, added a column for max minor allele frequency among gnomAD and TOPMed databases
    c_vrt_unnested = ['max_gnomad_topmed']
    c_vrt_nested = ["topmed", 'gnomad_genomes_2_1', 'gnomad_exomes_2_1', 'gnomad_genomes_3_0',
                    'gnomad_genomes_3_1_1']
    # c_vrt = ['max_gnomad_topmed', 'topmed', 'gnomad_genomes_2_1', 'gnomad_exomes_2_1', 'gnomad_genomes_3_0', 'gnomad_genomes_3_1_1']
    t_vrt = variants \
        .withColumn('max_gnomad_topmed',
                    F.greatest(F.lit(0), F.col('topmed')['af'], F.col('gnomad_exomes_2_1')['af'], \
                               F.col('gnomad_genomes_2_1')['af'], F.col('gnomad_genomes_3_0')['af'],
                               F.col('gnomad_genomes_3_1_1')['af'])) \
        .where((F.size(F.array_intersect(F.col('studies'), F.lit(F.array(*map(F.lit, study_id_list))))) > 0) & \
               F.col('chromosome').isin(chr_list)) \
        .select(cond + c_vrt_unnested + [F.col(nc + '.' + c).alias(nc + '_' + c)
                                         for nc in c_vrt_nested
                                         for c in variants.select(nc + '.*').columns])

    # Table ClinVar, restricted to those seen in variants and labeled as pathogenic/likely_pathogenic
    c_clv = ['VariationID', 'clin_sig']
    t_clv = clinvar \
        .withColumnRenamed('name', 'VariationID') \
        .where(F.split(F.split(F.col('geneinfo'), '\\|')[0], ':')[0].isin(gene_symbols_trunc) \
               & (F.array_contains(F.col('clin_sig'), 'Pathogenic') \
                  | F.array_contains(F.col('clin_sig'), 'Likely_pathogenic'))) \
        .join(t_vrt, cond) \
        .select(cond + c_clv)

    # Table ClinVar, restricted to those seen in variants and labeled as pathogenic/likely_pathogenic/vus
    # c_clv = ['VariationID', 'clin_sig']
    # t_clv = spark \
    #     .table('clinvar') \
    #     .withColumnRenamed('name', 'VariationID') \
    #     .where(F.split(F.split(F.col('geneinfo'), '\\|')[0], ':')[0].isin(gene_symbols_trunc) \
    #         & (F.array_contains(F.col('clin_sig'), 'Pathogenic') \
    #         | F.array_contains(F.col('clin_sig'), 'Likely_pathogenic') \
    #         | F.array_contains(F.col('clin_sig'), 'Uncertain_significance'))) \
    #     .join(t_vrt, cond) \
    #     .select(cond + c_clv)

    # Table HGMD, restricted to those seen in variants and labeled as DM or DM?
    c_hgmd = ['HGMDID', 'variant_class']
    t_hgmd = hg38_HGMD2022Q4_variant \
        .withColumnRenamed('id', 'HGMDID') \
        .where(F.col('symbol').isin(gene_symbols_trunc) \
               & F.col('variant_class').startswith('DM')) \
        .join(t_vrt, cond) \
        .select(cond + c_hgmd)

    # Join consequences, variants and dbnsfp, restricted to those with MAF less than a threshold and PredCountRatio_D2T within a range
    t_csq_vrt = t_csq \
        .join(t_vrt, cond)

    t_csq_vrt_dbn = t_csq_vrt \
        .join(t_dbn, cond, how='left') \
        .withColumn('flag', F.when( \
        (F.col('max_gnomad_topmed') < gnomAD_TOPMed_maf) \
        & (F.col('PredCountRatio_D2T').isNull() \
           | (F.col('PredCountRatio_D2T').isNotNull() \
              & (F.col('PredCountRatio_D2T') >= dpc_l) \
              & (F.col('PredCountRatio_D2T') <= dpc_u)) \
           ), 1) \
                    .otherwise(0))

    # Include ClinVar if specified
    if 'ClinVar' in known_variants_l and t_clv.count() > 0:
        t_csq_vrt_dbn = t_csq_vrt_dbn \
            .join(t_clv, cond, how='left') \
            .withColumn('flag', F.when(F.col('VariationID').isNotNull(), 1).otherwise(t_csq_vrt_dbn.flag))

    # Include HGMD if specified
    if 'HGMD' in known_variants_l and t_hgmd.count() > 0:
        t_csq_vrt_dbn = t_csq_vrt_dbn \
            .join(t_hgmd, cond, how='left') \
            .withColumn('flag', F.when(F.col('HGMDID').isNotNull(), 1).otherwise(t_csq_vrt_dbn.flag))

    # Table occurrences, restricted to input genes, chromosomes of those genes, input study IDs, and occurrences where alternate allele
    # is present, plus adjusted calls based on alternative allele fraction in the total sequencing depth
    c_ocr = ['ad', 'dp', 'calls', 'filter', 'is_lo_conf_denovo', 'is_hi_conf_denovo', 'is_proband',
             'affected_status', 'gender', \
             'biospecimen_id', 'participant_id', 'mother_id', 'father_id', 'family_id']
    t_ocr = occurrences \
        .withColumn('adjusted_calls',
                    F.when(F.col('ad')[1] / (F.col('ad')[0] + F.col('ad')[1]) < aaf, F.array(F.lit(0), F.lit(0))) \
                    .otherwise(F.col('calls'))) \
        .where(F.col('chromosome').isin(chr_list) \
               & (F.col('is_multi_allelic') == 'false') \
               & (F.col('has_alt') == 1) \
               & (F.col('adjusted_calls') != F.array(F.lit(0), F.lit(0)))) \
        .select(cond + c_ocr)

    # Finally join all together
    # t_output = F.broadcast(t_csq_vrt_dbn) \
    t_output = t_csq_vrt_dbn \
        .join(t_ocr, cond) \
        .where(t_csq_vrt_dbn.flag == 1) \
        .drop('flag')

    t_dgn = diagnoses \
        .select('participant_id', 'source_text_diagnosis') \
        .distinct() \
        .groupBy('participant_id') \
        .agg(F.collect_list('source_text_diagnosis').alias('diagnoses_combined'))
    t_pht = phenotypes \
        .select('participant_id', 'source_text_phenotype', 'hpo_id_phenotype') \
        .distinct() \
        .groupBy('participant_id') \
        .agg(F.collect_list('source_text_phenotype').alias('phenotypes_combined'), \
             F.collect_list('hpo_id_phenotype').alias('hpos_combined'))
    t_output_dgn_pht = t_output \
        .join(t_dgn, 'participant_id', 'left') \
        .join(t_pht, 'participant_id', 'left')

    return (t_output_dgn_pht)


# define output name and write table t_output
def write_output(t_output, gene_symbols_trunc, 
		        study_id_list, studies):
    short_code_id = list(studies \
                         .where(F.col('kf_id').isin(study_id_list)) \
                         .select('short_code').toPandas()['short_code'])
    Date = list(spark.sql("select current_date()") \
                .withColumn("current_date()", F.col("current_date()").cast("string")) \
                .toPandas()['current_date()'])
    output_filename= "_".join(Date + short_code_id + gene_symbols_trunc) +".tsv"
    t_output.toPandas() \
        .to_csv(output_filename, sep="\t", index=False, na_rep='-')
    # print("/sbgenomics/output-files/" + "_".join(Date + short_code_id + gene_symbols_trunc) +".tsv")

if args.hgmd is None:
	print("Missing hgmd parquet file", file=sys.stderr)
	exit(1)
if args.dbnsfp is None:
	print("Missing dbnsfp parquet file", file=sys.stderr)
	exit(1)
if args.clinvar is None:
	print("Missing clinvar parquet file", file=sys.stderr)
	exit(1)
if args.consequences is None:
	print("Missing consequences parquet file", file=sys.stderr)
	exit(1)
if args.variants is None:
	print("Missing variants parquet file", file=sys.stderr)
	exit(1)
if args.diagnoses is None:
	print("Missing diagnoses parquet file", file=sys.stderr)
	exit(1)
if args.phenotypes is None:
	print("Missing phenotypes parquet file", file=sys.stderr)
	exit(1)
if args.studies is None:
	print("Missing studies parquet file", file=sys.stderr)
	exit(1)
t_output = gene_based_filt(gene_symbols_trunc, study_id_list, gnomAD_TOPMed_maf, dpc_l, dpc_u,
		                known_variants_l, aaf, hg38_HGMD2022Q4_variant, dbnsfp_annovar, clinvar, 
	                    consequences, variants, diagnoses, phenotypes, occurrences)
write_output(t_output, gene_symbols_trunc, study_id_list, studies)