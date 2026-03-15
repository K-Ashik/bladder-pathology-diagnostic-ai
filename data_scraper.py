"""
Comprehensive PubMed Data Scraper
Downloads 400-500+ articles with expanded search queries
Optimized for 80%+ ML accuracy
"""

from Bio import Entrez
import pandas as pd
import time

Entrez.email = "research@diagnostic-ai.com"

print("=" * 80)
print("📚 COMPREHENSIVE PubMed Scraper - Target: 400-500 Articles")
print("=" * 80)
print()

def search_pubmed(query, max_results=50):
    """Search PubMed and return article IDs"""
    try:
        handle = Entrez.esearch(db="pubmed", term=query, retmax=max_results)
        record = Entrez.read(handle)
        handle.close()
        return record["IdList"]
    except Exception as e:
        print(f"❌ Error searching: {e}")
        return []

def fetch_article_details(pmid):
    """Fetch detailed information for a single article"""
    try:
        handle = Entrez.efetch(db="pubmed", id=pmid, rettype="xml")
        record = Entrez.read(handle)
        handle.close()
        
        article = record['PubmedArticle'][0]['MedlineCitation']['Article']
        
        title = article.get('ArticleTitle', 'N/A')
        
        abstract_text = article.get('Abstract', {})
        if isinstance(abstract_text, dict):
            abstract = ' '.join(abstract_text.get('AbstractText', ['N/A']))
        else:
            abstract = str(abstract_text)
        
        pubdate = article.get('Journal', {}).get('JournalIssue', {}).get('PubDate', {})
        year = pubdate.get('Year', 'N/A')
        
        journal = article.get('Journal', {}).get('Title', 'N/A')
        
        authors = article.get('AuthorList', [])
        first_author = authors[0].get('LastName', 'Unknown') if authors else 'Unknown'
        num_authors = len(authors)
        
        keywords = article.get('KeywordList', [])
        keywords_str = '; '.join([str(k) for k in keywords]) if keywords else 'N/A'
        
        return {
            'pmid': pmid,
            'title': title,
            'abstract': abstract,
            'year': year,
            'journal': journal,
            'first_author': first_author,
            'num_authors': num_authors,
            'keywords': keywords_str
        }
    
    except Exception as e:
        return None

# EXPANDED search queries - 35 queries for comprehensive coverage
search_queries = {
    # MALAKOPLAKIA - Core (8 queries)
    'malakoplakia_bladder': ('malakoplakia bladder', 25),
    'malakoplakia_urinary': ('malakoplakia urinary tract', 20),
    'malakoplakia_diagnosis': ('malakoplakia diagnosis pathology', 25),
    'malakoplakia_case_reports': ('malakoplakia case report', 25),
    'malakoplakia_atypical': ('malakoplakia atypical unusual', 20),
    'malakoplakia_michaelis': ('michaelis gutmann bodies', 20),
    'malakoplakia_von_hansemann': ('von hansemann histiocytes', 15),
    'malakoplakia_treatment': ('malakoplakia treatment management', 15),
    
    # SCHISTOSOMIASIS - Core (8 queries)
    'schistosomiasis_bladder': ('schistosomiasis haematobium bladder', 25),
    'schistosomiasis_diagnosis': ('schistosomiasis diagnosis pathology', 25),
    'schistosomiasis_eggs': ('schistosomiasis eggs parasitic', 20),
    'schistosomiasis_complications': ('schistosomiasis bladder complications', 20),
    'schistosomiasis_case_reports': ('schistosomiasis case report', 20),
    'schistosomiasis_granuloma': ('schistosomiasis granulomatous', 15),
    'schistosomiasis_imaging': ('schistosomiasis imaging cystoscopy', 15),
    'schistosomiasis_treatment': ('schistosomiasis treatment management', 15),
    
    # GRANULOMATOUS INFLAMMATION (5 queries)
    'granulomatous_cystitis': ('granulomatous cystitis diagnosis', 20),
    'granuloma_bladder': ('granuloma bladder pathology', 20),
    'granulomatous_inflammation': ('granulomatous inflammation urinary', 15),
    'granulomatous_lesions': ('granulomatous lesions differential', 15),
    'granulomatous_urethritis': ('granulomatous urethritis cystitis', 12),
    
    # DIFFERENTIAL DIAGNOSIS (6 queries)
    'bladder_differential': ('bladder biopsy differential diagnosis', 25),
    'bladder_pathology': ('bladder pathology histology', 25),
    'atypical_bladder': ('atypical bladder lesions diagnosis', 20),
    'bladder_mimics': ('bladder cancer mimics pathology', 15),
    'urinary_lesions': ('urinary tract lesions differential', 15),
    'bladder_infection': ('bladder infection granulomas', 12),
    
    # PARASITIC INFECTIONS (3 queries)
    'parasitic_bladder': ('parasitic infection bladder diagnosis', 20),
    'parasitic_cystitis': ('parasitic cystitis diagnosis', 15),
    'helminthic_infection': ('helminthic infection urinary tract', 12),
    
    # CLINICAL PRESENTATION (4 queries)
    'bladder_ulceration': ('bladder ulceration diagnosis', 15),
    'hematuria_investigation': ('hematuria bladder diagnosis', 20),
    'bladder_fibrosis': ('bladder fibrosis calcification', 15),
    'chronic_cystitis': ('chronic cystitis granulomatous', 15),
    
    # IMAGING & CYSTOSCOPY (2 queries)
    'cystoscopy_findings': ('cystoscopy bladder pathology', 15),
    'bladder_imaging': ('bladder imaging ultrasound CT', 12),
}

all_articles = []
total_queries = len(search_queries)

print(f"🔍 Searching across {total_queries} comprehensive queries")
print(f"📊 Target: 400-500 unique articles")
print(f"⏱️  Estimated time: 10-15 minutes")
print("=" * 80)
print()

articles_per_query = []

for query_idx, (query_name, (query_text, max_per_query)) in enumerate(search_queries.items(), 1):
    print(f"[{query_idx:2d}/{total_queries}] 📖 {query_name}")
    print(f"        Query: \"{query_text}\"")
    
    pmids = search_pubmed(query_text, max_results=max_per_query)
    print(f"        Found: {len(pmids)} articles")
    
    query_articles = 0
    for article_idx, pmid in enumerate(pmids, 1):
        article = fetch_article_details(pmid)
        
        if article:
            article['source_query'] = query_name
            all_articles.append(article)
            query_articles += 1
            print(f"        ✅ [{article_idx:2d}/{len(pmids)}]", end="")
            print(f" - Total so far: {len(all_articles)}")
        else:
            print(f"        ❌ [{article_idx:2d}/{len(pmids)}]")
        
        time.sleep(0.2)
    
    articles_per_query.append({
        'query': query_name,
        'articles_found': query_articles
    })
    
    print()
    time.sleep(0.5)

print()
print("=" * 80)
print("✅ DATA COLLECTION COMPLETE")
print("=" * 80)
print()

# Remove duplicates
unique_pmids = {}
for article in all_articles:
    pmid = article['pmid']
    if pmid not in unique_pmids:
        unique_pmids[pmid] = article

final_articles = list(unique_pmids.values())

# Create dataframe
df = pd.DataFrame(final_articles)

# Save to CSV
df.to_csv("dataset.csv", index=False)

print(f"📈 FINAL STATISTICS:")
print(f"   Total articles fetched: {len(all_articles)}")
print(f"   Unique articles (after dedup): {len(df)}")
print(f"   Duplicates removed: {len(all_articles) - len(df)}")
print()

print(f"📊 BREAKDOWN BY CATEGORY:")
category_summary = {
    'Malakoplakia': sum(item['articles_found'] for item in articles_per_query if 'malakoplakia' in item['query']),
    'Schistosomiasis': sum(item['articles_found'] for item in articles_per_query if 'schistosomiasis' in item['query']),
    'Granulomatous': sum(item['articles_found'] for item in articles_per_query if 'granulomatous' in item['query']),
    'Differential': sum(item['articles_found'] for item in articles_per_query if 'differential' in item['query'] or 'bladder_pathology' in item['query']),
    'Parasitic': sum(item['articles_found'] for item in articles_per_query if 'parasitic' in item['query']),
    'Clinical': sum(item['articles_found'] for item in articles_per_query if 'ulceration' in item['query'] or 'hematuria' in item['query'] or 'cystitis' in item['query']),
    'Imaging': sum(item['articles_found'] for item in articles_per_query if 'cystoscopy' in item['query'] or 'imaging' in item['query']),
}

for category, count in category_summary.items():
    print(f"   {category:20s}: {count:3d} articles")
print()

print(f"📋 DATASET INFO:")
print(f"   Columns: {list(df.columns)}")
print(f"   Shape: {df.shape}")
print()

print(f"📅 PUBLICATION YEARS:")
print(f"   From: {df['year'].min()}")
print(f"   To: {df['year'].max()}")
print()

print(f"✍️  AUTHOR STATISTICS:")
print(f"   Average authors per article: {df['num_authors'].mean():.1f}")
print(f"   Max authors: {df['num_authors'].max()}")
print(f"   Min authors: {df['num_authors'].min()}")
print()

print(f"🏥 TOP JOURNALS:")
print(df['journal'].value_counts().head(15))
print()

print("=" * 80)
print("✅ DATASET READY FOR FEATURE EXTRACTION")
print("=" * 80)
print(f"File: dataset.csv")
print(f"Total articles: {len(df)}")
print()