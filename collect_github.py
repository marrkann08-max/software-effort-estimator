"""
collect_github.py  —  collect GitHub project metrics and derive effort proxy
Run once: python collect_github.py
Saves: github_projects.csv
"""
import requests, time, os, json
import pandas as pd
import numpy as np
from datetime import datetime

TOKEN = os.environ.get("GITHUB_TOKEN", "")
HEADERS = {
    "Authorization": f"token {TOKEN}",
    "Accept": "application/vnd.github.v3+json",
}
BASE    = "https://api.github.com"
SINCE   = "2021-01-01T00:00:00Z"   # Modern-development sampling window
COMMIT_MAX_PAGES = 12
COMMITS_PER_PAGE = 100
COMMIT_SAMPLE_LIMIT = COMMIT_MAX_PAGES * COMMITS_PER_PAGE

# ── Curated repos: medium-sized, team projects, active PR workflow ─────────────
REPOS = [
    # ── Python web / async ──────────────────────────────────────────────────
    "psf/requests",
    "pallets/flask",
    "tiangolo/fastapi",
    "pallets/click",
    "Textualize/rich",
    "encode/httpx",
    "pydantic/pydantic",
    "pytest-dev/pytest",
    "celery/celery",
    "scrapy/scrapy",
    "encode/django-rest-framework",
    "encode/starlette",
    "encode/uvicorn",
    "aiohttp/aiohttp",
    "tornadoweb/tornado",
    "falconry/falcon",
    "pallets/werkzeug",
    # ── Python data / ML ────────────────────────────────────────────────────
    "huggingface/datasets",
    "streamlit/streamlit",
    "plotly/plotly.py",
    "numpy/numpy",
    "matplotlib/matplotlib",
    "scikit-learn/scikit-learn",
    "Lightning-AI/lightning",
    "mlflow/mlflow",
    "prefecthq/prefect",
    "pola-rs/polars",
    # ── Python tooling ──────────────────────────────────────────────────────
    "python-poetry/poetry",
    "pre-commit/pre-commit",
    "psf/black",
    "astral-sh/ruff",
    "pypa/pip",
    "sphinx-doc/sphinx",
    "mkdocs/mkdocs",
    "Delgan/loguru",
    "tiangolo/typer",
    "httpie/cli",
    "gitpython-developers/GitPython",
    "beetbox/beets",
    # ── JavaScript / TypeScript ─────────────────────────────────────────────
    "expressjs/express",
    "axios/axios",
    "vitejs/vite",
    "nestjs/nest",
    "sveltejs/svelte",
    "trpc/trpc",
    "prisma/prisma",
    # ── Go ──────────────────────────────────────────────────────────────────
    "gin-gonic/gin",
    "gofiber/fiber",
    "cli/cli",
    "helm/helm",
    "prometheus/prometheus",
    "grafana/loki",
    "charmbracelet/bubbletea",
    # ── Rust ────────────────────────────────────────────────────────────────
    "BurntSushi/ripgrep",
    "sharkdp/bat",
    "tokio-rs/axum",
    # ── Ruby / Java ─────────────────────────────────────────────────────────
    "jekyll/jekyll",
    "square/okhttp",
    # ── Observability / DevOps ──────────────────────────────────────────────
    "docker/compose",
    "open-telemetry/opentelemetry-python",
    # ── Large multi-contributor ─────────────────────────────────────────────
    "home-assistant/core",
    "sqlalchemy/sqlalchemy",
    # ── Additional Python ───────────────────────────────────────────────────
    "tiangolo/sqlmodel",
    "astral-sh/uv",
    "encode/anyio",
    "python-attrs/attrs",
    "pallets/jinja",
    "pypa/virtualenv",
    "jazzband/django-debug-toolbar",
    "redis/redis-py",
    "encode/databases",
    "graphene-python/graphene",
    "strawberry-graphql/strawberry",
    "simonw/datasette",
    "httpie/cli",
    # ── Additional JavaScript / TypeScript ──────────────────────────────────
    "TanStack/query",
    "TanStack/router",
    "pmndrs/zustand",
    "immerjs/immer",
    "biomejs/biome",
    "evanw/esbuild",
    "radix-ui/primitives",
    # ── Additional Go ───────────────────────────────────────────────────────
    "go-chi/chi",
    "labstack/echo",
    "go-gorm/gorm",
    "spf13/cobra",
    "spf13/viper",
    "urfave/cli",
    "kubernetes-sigs/kind",
    # ── Additional Rust ─────────────────────────────────────────────────────
    "serde-rs/serde",
    "clap-rs/clap",
    "actix/actix-web",
    "rayon-rs/rayon",
    # ── Additional Ruby / other ─────────────────────────────────────────────
    "sinatra/sinatra",
    "jekyll/jekyll",
    # ── DevOps / infra ──────────────────────────────────────────────────────
    "hashicorp/vault",
    "kubernetes-sigs/kustomize",
    "argoproj/argo-cd",
]


def get_api(url, params=None):
    for attempt in range(4):
        try:
            r = requests.get(url, headers=HEADERS, params=params, timeout=20)
        except requests.RequestException as e:
            print(f"  network error: {e}, retrying...")
            time.sleep(5)
            continue
        if r.status_code == 202:
            time.sleep(4)
            continue
        if r.status_code == 403:
            reset = int(r.headers.get("X-RateLimit-Reset", time.time() + 65))
            wait  = max(reset - time.time() + 5, 5)
            print(f"  rate limit — waiting {wait:.0f}s")
            time.sleep(wait)
            continue
        return r
    return None


def paginate(url, params=None, max_pages=8):
    p = dict(params or {})
    p["per_page"] = 100
    out = []
    for page in range(1, max_pages + 1):
        p["page"] = page
        r = get_api(url, p)
        if r is None or r.status_code != 200:
            break
        data = r.json()
        if not data:
            break
        out.extend(data)
        if len(data) < 100:
            break
    return out


def collect(owner_repo):
    owner, repo = owner_repo.split("/")
    print(f"\n> {owner_repo}")

    # ── 1. Repo metadata ───────────────────────────────────────────────────────
    r = get_api(f"{BASE}/repos/{owner}/{repo}")
    if r is None or r.status_code != 200:
        print(f"  SKIP — {r.status_code if r else 'no response'}"); return None
    meta = r.json()

    # ── 2. Languages ───────────────────────────────────────────────────────────
    r_lang = get_api(f"{BASE}/repos/{owner}/{repo}/languages")
    lang_count = len(r_lang.json()) if (r_lang and r_lang.status_code == 200) else 1

    # ── 3. CI presence ─────────────────────────────────────────────────────────
    r_ci = get_api(f"{BASE}/repos/{owner}/{repo}/contents/.github/workflows")
    has_ci = 1 if (r_ci and r_ci.status_code == 200) else 0

    # ── 4. Test file ratio from git tree (accurate count, 1 API call) ───────────
    test_file_ratio = 0.10
    r_tree = get_api(f"{BASE}/repos/{owner}/{repo}/git/trees/HEAD",
                     params={"recursive": "1"})
    if r_tree and r_tree.status_code == 200:
        tree_files = [item["path"] for item in r_tree.json().get("tree", [])
                      if item["type"] == "blob"]
        if tree_files:
            total = len(tree_files)
            test_n = sum(1 for p in tree_files if
                         "/test" in p.lower() or "/tests" in p.lower() or
                         "/spec"  in p.lower() or "test_"  in p.lower() or
                         "_test." in p.lower() or "_spec." in p.lower())
            test_file_ratio = round(test_n / total, 4)

    # ── 5. Commits since 2021 ─────────────────────────────────────────────────
    commits = paginate(
        f"{BASE}/repos/{owner}/{repo}/commits",
        {"since": SINCE},
        max_pages=COMMIT_MAX_PAGES,
    )
    print(f"  commits: {len(commits)}")
    commit_sample_limit_hit = len(commits) >= COMMIT_SAMPLE_LIMIT
    if commit_sample_limit_hit:
        print(
            f"  WARNING - commit history reached the {COMMIT_SAMPLE_LIMIT}-commit "
            "sampling ceiling; author-months is censored near 60"
        )
    if len(commits) < 15:
        print("  SKIP — too few commits"); return None

    author_days = set()
    for c in commits:
        try:
            if c.get("author") and c["author"].get("login"):
                author = c["author"]["login"]
            else:
                author = c["commit"]["author"]["email"]
            date = c["commit"]["author"]["date"][:10]
            author_days.add((author, date))
        except Exception:
            continue

    author_months    = len(author_days) / 20
    contributor_count = len({a for a, _ in author_days})

    # commit frequency (commits / week)
    dates = []
    try:
        dates  = [c["commit"]["author"]["date"][:10] for c in commits]
        d0, d1 = min(dates), max(dates)
        weeks  = max((datetime.fromisoformat(d1) - datetime.fromisoformat(d0)).days / 7, 1)
        commit_freq = len(commits) / weeks
    except Exception:
        commit_freq = 1.0

    # ── 6. Closed PRs ─────────────────────────────────────────────────────────
    prs = paginate(f"{BASE}/repos/{owner}/{repo}/pulls",
                   {"state": "closed", "sort": "updated", "direction": "desc"},
                   max_pages=4)
    print(f"  PRs: {len(prs)}")

    merged      = [p for p in prs if p.get("merged_at")]
    pr_merge_rate = len(merged) / max(len(prs), 1)

    review_days = []
    for p in merged[:60]:
        try:
            c = datetime.fromisoformat(p["created_at"].replace("Z", "+00:00"))
            m = datetime.fromisoformat(p["merged_at"].replace("Z",  "+00:00"))
            review_days.append((m - c).total_seconds() / 86400)
        except Exception:
            continue

    avg_pr_review_days = float(np.median(review_days)) if review_days else 3.0

    # KLOC proxy: repo disk size (GitHub reports in KB)
    repo_size_kb = meta.get("size", 1000)
    total_kloc   = max(repo_size_kb / 50, 1.0)   # rough: 50 KB ≈ 1 KLOC

    row = {
        "repo":               owner_repo,
        "author_months":      round(author_months, 2),
        "total_kloc":         round(total_kloc, 1),
        "contributor_count":  contributor_count,
        "avg_pr_review_days": round(avg_pr_review_days, 2),
        "pr_merge_rate":      round(pr_merge_rate, 3),
        "has_ci":             has_ci,
        "test_file_ratio":    round(test_file_ratio, 3),
        "lang_count":         lang_count,
        "commit_frequency":   round(commit_freq, 2),
        "stars":              meta.get("stargazers_count", 0),
        "repo_created_at":    meta.get("created_at", ""),
        "first_sampled_commit_at": min(dates) if dates else "",
        "last_sampled_commit_at":  max(dates) if dates else "",
        "sampled_commit_count":    len(commits),
        "commit_sample_limit":     COMMIT_SAMPLE_LIMIT,
        "commit_sample_limit_hit": commit_sample_limit_hit,
    }
    print(f"  effort={author_months:.1f} PM  contributors={contributor_count}  "
          f"KLOC={total_kloc:.0f}  review={avg_pr_review_days:.1f}d")
    return row


# ── Main ───────────────────────────────────────────────────────────────────────
csv_out = os.path.join(os.path.dirname(__file__), "github_projects.csv")

# Load existing data — skip repos already collected
existing_repos = set()
rows = []
if os.path.exists(csv_out):
    existing_df = pd.read_csv(csv_out)
    rows = existing_df.to_dict("records")
    existing_repos = set(existing_df["repo"].tolist())
    print(f"Loaded {len(rows)} existing repos from {csv_out}")

todo = [r for r in REPOS if r not in existing_repos]
print(f"Collecting {len(todo)} new repos (since {SINCE[:10]})...")

r_check = get_api(f"{BASE}/rate_limit")
if r_check and r_check.status_code == 200:
    rl = r_check.json()["rate"]
    print(f"Rate limit: {rl['remaining']}/{rl['limit']} remaining, "
          f"resets at {datetime.fromtimestamp(rl['reset']).strftime('%H:%M:%S')}\n")

for repo in todo:
    row = collect(repo)
    if row:
        rows.append(row)
        pd.DataFrame(rows).to_csv(csv_out, index=False)
        print(f"  saved ({len(rows)} total repos)")
    time.sleep(0.5)

df = pd.DataFrame(rows)
print(f"\n{'='*50}")
print(f"Collected {len(df)} repos")
print(df[["repo","author_months","contributor_count","total_kloc","avg_pr_review_days"]].to_string(index=False))
print(f"\nSaved: github_projects.csv")
print(f"\nFeature correlations with author_months (effort):")
num_cols = ["total_kloc","contributor_count","avg_pr_review_days","pr_merge_rate",
            "has_ci","test_file_ratio","lang_count","commit_frequency"]
for col in num_cols:
    corr = df["author_months"].corr(df[col])
    print(f"  {col:>22s}: r={corr:+.3f}")
