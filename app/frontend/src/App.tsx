import { useEffect, useRef, useState, type ChangeEvent, type FormEvent } from "react";
import { getRecommendations, swapCard, checkHealth } from "./api";
import type {
  Modality,
  ModelName,
  ProductCard,
  QueryProfile,
  RecommendResponse,
} from "./types";

const MODALITIES: Modality[] = ["book", "film", "music", "writing"];
const MODALITY_LABEL: Record<Modality, string> = {
  book: "Book",
  film: "Film",
  music: "Music",
  writing: "Writing",
};

const PLACEHOLDERS = [
  "describe how you want to feel tonight…",
  "paste a line of poetry that won't leave your head…",
  "what are you trying to escape from right now?",
  "music for studying after a breakup",
  "the aesthetic of a late-night convenience store",
  "something quiet and slow, like rain on glass",
];

function randomPlaceholder() {
  return PLACEHOLDERS[Math.floor(Math.random() * PLACEHOLDERS.length)];
}

export default function App() {
  const [query, setQuery] = useState("");
  const [placeholder] = useState(randomPlaceholder());
  const [modalities, setModalities] = useState<Set<Modality>>(
    new Set(MODALITIES),
  );
  const [model, setModel] = useState<ModelName>("two_tower");
  const [loading, setLoading] = useState(false);
  const [result, setResult] = useState<RecommendResponse | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [backendUp, setBackendUp] = useState<boolean | null>(null);
  const [openPoem, setOpenPoem] = useState<ProductCard | null>(null);
  const [imageFile, setImageFile] = useState<
    { dataUrl: string; name: string; size: number } | null
  >(null);
  const [imageError, setImageError] = useState<string | null>(null);
  const imageInputRef = useRef<HTMLInputElement>(null);
  // Per-modality "already shown" set of item ids; used as exclude list when
  // swapping so we never loop back to a card the user just rejected.
  const [excludedIds, setExcludedIds] = useState<Record<string, string[]>>({});
  const [swappingModality, setSwappingModality] = useState<string | null>(null);

  useEffect(() => {
    let alive = true;
    checkHealth().then((ok) => {
      if (alive) setBackendUp(ok);
    });
    return () => {
      alive = false;
    };
  }, []);

  const toggleModality = (m: Modality) => {
    const next = new Set(modalities);
    if (next.has(m)) next.delete(m);
    else next.add(m);
    setModalities(next);
  };

  async function handleSubmit(e: FormEvent) {
    e.preventDefault();
    if ((!query.trim() && !imageFile) || loading) return;
    setLoading(true);
    setError(null);
    setResult(null);
    try {
      const resp = await getRecommendations({
        query: query.trim(),
        modalities: Array.from(modalities),
        model,
        image_base64: imageFile?.dataUrl,
      });
      setResult(resp);
      // Seed the excluded-ids map with whatever came back, so the first swap
      // on any modality already knows to skip the currently-displayed card.
      const seeded: Record<string, string[]> = {};
      for (const [m, cards] of Object.entries(resp.results)) {
        seeded[m] = cards.map((c) => c.id);
      }
      setExcludedIds(seeded);
    } catch (err) {
      setError(err instanceof Error ? err.message : String(err));
    } finally {
      setLoading(false);
    }
  }

  async function handleSwap(m: Modality) {
    if (!result || swappingModality) return;
    setSwappingModality(m);
    setError(null);
    try {
      const exclude = excludedIds[m] ?? [];
      const resp = await swapCard({
        query: query.trim(),
        modalities: Array.from(modalities),
        model,
        image_base64: imageFile?.dataUrl,
        swap_modality: m,
        exclude_ids: exclude,
      });
      setResult(resp);
      // The swapped card's id joins the modality's exclude list for future swaps.
      const newCards = resp.results[m] ?? [];
      setExcludedIds((prev) => ({
        ...prev,
        [m]: Array.from(new Set([...(prev[m] ?? []), ...newCards.map((c) => c.id)])),
      }));
    } catch (err) {
      setError(err instanceof Error ? err.message : String(err));
    } finally {
      setSwappingModality(null);
    }
  }

  function handleImageChange(e: ChangeEvent<HTMLInputElement>) {
    const file = e.target.files?.[0];
    // Reset the native input so the same file can be re-picked later.
    if (imageInputRef.current) imageInputRef.current.value = "";
    if (!file) return;
    setImageError(null);
    if (!file.type.startsWith("image/")) {
      setImageError("File must be an image.");
      return;
    }
    const MAX = 5 * 1024 * 1024;
    if (file.size > MAX) {
      setImageError(
        `Image is ${(file.size / 1024 / 1024).toFixed(1)}MB; max is 5MB.`,
      );
      return;
    }
    const reader = new FileReader();
    reader.onload = () => {
      setImageFile({
        dataUrl: String(reader.result),
        name: file.name,
        size: file.size,
      });
      // New image invalidates the currently-shown result — require re-submit.
      setResult(null);
    };
    reader.onerror = () => setImageError("Failed to read image file.");
    reader.readAsDataURL(file);
  }

  function clearImage() {
    setImageFile(null);
    setImageError(null);
    setResult(null);
  }


  return (
    <div className="min-h-screen">
      <div className="mx-auto max-w-5xl px-6 py-10">
        <header className="mb-10 flex items-start justify-between">
          <div>
            <h1 className="font-serif text-6xl leading-none text-ink">Wave</h1>
            <p className="mt-2 max-w-md text-sm text-ink/60">
              Describe a feeling. Get a book, a film, a song, and a piece of
              writing that share the same aesthetic.
            </p>
          </div>
          <BackendIndicator up={backendUp} />
        </header>

        <form onSubmit={handleSubmit} className="space-y-5">
          <textarea
            value={query}
            onChange={(e) => setQuery(e.target.value)}
            placeholder={placeholder}
            rows={3}
            className="w-full resize-none rounded-md border border-muted bg-white p-5 font-serif text-2xl leading-relaxed text-ink placeholder-ink/30 shadow-sm transition focus:border-accent focus:outline-none focus:ring-1 focus:ring-accent/40"
            disabled={loading}
          />

          <ImageUploader
            imageFile={imageFile}
            imageError={imageError}
            onPick={handleImageChange}
            onClear={clearImage}
            disabled={loading}
            inputRef={imageInputRef}
          />

          <div className="flex flex-wrap items-center gap-x-6 gap-y-3 text-sm">
            <div className="flex flex-wrap items-center gap-3">
              <span className="text-ink/50">include</span>
              {MODALITIES.map((m) => (
                <label
                  key={m}
                  className="flex cursor-pointer items-center gap-1.5 select-none"
                >
                  <input
                    type="checkbox"
                    checked={modalities.has(m)}
                    onChange={() => {
                      toggleModality(m);
                      setResult(null);
                    }}
                    className="accent-accent"
                    disabled={loading}
                  />
                  <span>{MODALITY_LABEL[m]}</span>
                </label>
              ))}
            </div>

            <div className="flex items-center gap-2">
              <span className="text-ink/50">model</span>
              <select
                value={model}
                onChange={(e) => {
                  setModel(e.target.value as ModelName);
                  setResult(null);
                }}
                disabled={loading}
                className="rounded border border-muted bg-white px-2 py-1 focus:border-accent focus:outline-none"
              >
                <option value="two_tower">two_tower (final)</option>
                <option value="knn">knn (classical ML)</option>
                <option value="popularity">popularity (baseline)</option>
              </select>
            </div>

            <button
              type="submit"
              disabled={
                (!query.trim() && !imageFile) ||
                loading ||
                modalities.size === 0
              }
              className="ml-auto rounded bg-accent px-6 py-2 text-white shadow-sm transition hover:bg-accent/90 disabled:cursor-not-allowed disabled:bg-muted disabled:text-ink/40 disabled:shadow-none"
            >
              {loading ? "thinking…" : "submit"}
            </button>
          </div>
        </form>

        {error && (
          <div className="mt-8 rounded border-l-4 border-red-500 bg-red-50 p-4 text-sm text-red-700">
            <strong className="font-semibold">error:</strong> {error}
          </div>
        )}

        {loading && <LoadingStrip />}

        {result && !loading && (
          <Results
            result={result}
            onOpenPoem={setOpenPoem}
            onSwap={handleSwap}
            swappingModality={swappingModality}
          />
        )}

        {!result && !loading && !error && <HintsList />}
      </div>

      {openPoem && (
        <PoemModal card={openPoem} onClose={() => setOpenPoem(null)} />
      )}
    </div>
  );
}

function ImageUploader({
  imageFile,
  imageError,
  onPick,
  onClear,
  disabled,
  inputRef,
}: {
  imageFile: { dataUrl: string; name: string; size: number } | null;
  imageError: string | null;
  onPick: (e: ChangeEvent<HTMLInputElement>) => void;
  onClear: () => void;
  disabled: boolean;
  inputRef: React.RefObject<HTMLInputElement>;
}) {
  return (
    <div className="flex flex-col gap-2">
      <input
        ref={inputRef}
        id="wave-image-upload"
        type="file"
        accept="image/*"
        onChange={onPick}
        disabled={disabled}
        className="hidden"
      />
      {imageFile ? (
        <div className="relative w-fit max-w-full">
          <img
            src={imageFile.dataUrl}
            alt=""
            className="block max-h-[50vh] max-w-full rounded border border-muted bg-white object-contain shadow-sm"
          />
          <button
            type="button"
            onClick={onClear}
            disabled={disabled}
            aria-label="remove image"
            className="absolute right-2 top-2 flex h-8 w-8 items-center justify-center rounded-full border border-muted bg-white/95 text-xl leading-none text-ink/60 shadow-sm transition hover:border-red-400 hover:text-red-500 disabled:opacity-40"
          >
            ×
          </button>
        </div>
      ) : (
        <label
          htmlFor="wave-image-upload"
          className={`inline-flex w-fit cursor-pointer items-center gap-2 rounded border border-dashed border-muted px-3 py-2 text-sm text-ink/60 transition hover:border-accent hover:text-accent ${
            disabled ? "pointer-events-none opacity-50" : ""
          }`}
        >
          <span>📷</span>
          <span>add an image (optional, ≤5MB)</span>
        </label>
      )}
      {imageError && (
        <p className="text-xs text-red-600">{imageError}</p>
      )}
    </div>
  );
}

function BackendIndicator({ up }: { up: boolean | null }) {
  const color =
    up === null
      ? "bg-muted"
      : up
        ? "bg-emerald-500"
        : "bg-red-500";
  const label = up === null ? "checking…" : up ? "backend up" : "backend down";
  return (
    <div className="flex items-center gap-2 text-xs text-ink/50">
      <span className={`h-2 w-2 rounded-full ${color}`} />
      {label}
    </div>
  );
}

function LoadingStrip() {
  return (
    <div className="mt-12 flex items-center gap-3 text-ink/50">
      <span className="inline-block h-2 w-2 animate-pulse rounded-full bg-accent" />
      <span className="font-serif italic">searching the vibe…</span>
    </div>
  );
}

function HintsList() {
  return (
    <div className="mt-16 space-y-2 text-sm text-ink/40">
      <p className="font-serif italic">try asking for…</p>
      <ul className="ml-4 list-disc space-y-1">
        <li>"something I can fall asleep thinking about"</li>
        <li>"warmth without sentimentality"</li>
        <li>"energetic but not loud"</li>
        <li>"remind me why life is worth it"</li>
      </ul>
    </div>
  );
}

function Results({
  result,
  onOpenPoem,
  onSwap,
  swappingModality,
}: {
  result: RecommendResponse;
  onOpenPoem: (card: ProductCard) => void;
  onSwap: (m: Modality) => void;
  swappingModality: string | null;
}) {
  const { query_profile, results } = result;
  const ordered = MODALITIES.filter(
    (m) => results[m] && results[m].length > 0,
  );

  const lgCols: Record<number, string> = {
    1: "lg:grid-cols-1",
    2: "lg:grid-cols-2",
    3: "lg:grid-cols-3",
    4: "lg:grid-cols-4",
  };
  const lgColClass = lgCols[ordered.length] ?? "lg:grid-cols-4";

  return (
    <div className="mt-10 space-y-8">
      <ProfileBar profile={query_profile} />
      <div className={`grid grid-cols-1 gap-5 md:grid-cols-2 ${lgColClass}`}>
        {ordered.map((m) => (
          <div key={m}>
            <div className="mb-2 text-[10px] uppercase tracking-widest text-ink/40">
              {MODALITY_LABEL[m as Modality]}
            </div>
            {results[m].map((card) => (
              <Card
                key={card.id}
                card={card}
                onOpenPoem={onOpenPoem}
                onSwap={() => onSwap(m as Modality)}
                swapping={swappingModality === m}
              />
            ))}
          </div>
        ))}
      </div>
    </div>
  );
}

function ProfileBar({ profile }: { profile: QueryProfile }) {
  return (
    <div className="rounded-md border border-muted bg-white/70 p-5">
      <p className="mb-3 font-serif text-lg italic leading-relaxed text-ink/75">
        "{profile.vibe_summary}"
      </p>
      <div className="flex flex-wrap gap-2">
        {profile.aesthetic_tags.map((t) => (
          <span
            key={t}
            className="rounded-full bg-muted px-3 py-0.5 text-xs text-ink/70"
          >
            {t}
          </span>
        ))}
      </div>
    </div>
  );
}

function Card({
  card,
  onOpenPoem,
  onSwap,
  swapping,
}: {
  card: ProductCard;
  onOpenPoem: (c: ProductCard) => void;
  onSwap: () => void;
  swapping: boolean;
}) {
  const [imgErr, setImgErr] = useState(false);
  const isPoem = card.subtype === "poem";
  // Some Goodreads / Gutenberg URLs serve a branded "no photo" asset that
  // loads with HTTP 200 (so onError never fires). Detect those by URL pattern
  // and treat them as missing, so TitlePoster kicks in.
  const isPlaceholderUrl = (url: string) => {
    const u = (url || "").toLowerCase();
    return (
      u.includes("nophoto") ||
      u.includes("no_image") ||
      u.includes("no-image") ||
      u.includes("noimage")
    );
  };
  const hasImg =
    !!card.cover_url && !imgErr && !isPlaceholderUrl(card.cover_url);

  const hasExcerpt = (card.excerpt ?? "").trim().length > 0;
  const cover =
    isPoem && hasExcerpt ? (
      <PoemPoster excerpt={card.excerpt ?? ""} />
    ) : hasImg ? (
      <img
        src={card.cover_url}
        alt={card.title}
        loading="lazy"
        onError={() => setImgErr(true)}
        className="h-full w-full object-cover"
      />
    ) : (
      <TitlePoster card={card} />
    );

  const body = (
    <>
      <div className="relative aspect-[3/4] w-full bg-muted">
        {cover}
        <button
          type="button"
          onClick={(e) => {
            e.preventDefault();
            e.stopPropagation();
            if (!swapping) onSwap();
          }}
          disabled={swapping}
          aria-label={`swap ${card.modality} recommendation`}
          title="swap for another"
          className={`absolute right-2 top-2 flex h-8 w-8 items-center justify-center rounded-full border border-muted bg-white/90 text-base leading-none text-ink/60 shadow-sm transition hover:border-accent hover:text-accent disabled:opacity-60 ${
            swapping ? "animate-spin" : ""
          }`}
        >
          ↻
        </button>
      </div>
      <div className="space-y-1 p-4">
        <h3 className="font-serif text-lg leading-tight text-ink group-hover:text-accent">
          {card.title}
        </h3>
        {(card.creator || card.year > 0) && (
          <p className="text-xs text-ink/60">
            {card.creator}
            {card.creator && card.year > 0 ? " · " : ""}
            {card.year > 0 ? card.year : ""}
          </p>
        )}
        {card.why_this && (
          <p className="pt-2 font-serif text-sm italic leading-snug text-ink/75">
            {card.why_this}
          </p>
        )}
        <p className="pt-2 text-[10px] uppercase tracking-wider text-ink/35">
          sim {card.similarity.toFixed(3)}
        </p>
      </div>
    </>
  );

  const shell =
    "group block overflow-hidden rounded-md border border-muted bg-white transition hover:-translate-y-0.5 hover:border-accent hover:shadow-md";

  if (isPoem) {
    return (
      <div
        role="button"
        tabIndex={0}
        onClick={() => onOpenPoem(card)}
        onKeyDown={(e) => {
          if (e.key === "Enter" || e.key === " ") {
            e.preventDefault();
            onOpenPoem(card);
          }
        }}
        className={`${shell} cursor-pointer`}
      >
        {body}
      </div>
    );
  }

  return (
    <a
      href={card.external_url || "#"}
      target="_blank"
      rel="noopener noreferrer"
      className={shell}
    >
      {body}
    </a>
  );
}

function TitlePoster({ card }: { card: ProductCard }) {
  return (
    <div
      className="flex h-full w-full flex-col justify-center overflow-hidden px-5 py-6 text-center"
      style={{
        backgroundColor: "#EFE8D4",
        color: "#3a332b",
        boxShadow: "inset 0 0 40px rgba(80, 60, 40, 0.08)",
      }}
    >
      <p
        className="mb-3 text-[10px] uppercase"
        style={{
          color: "#8B6F47",
          letterSpacing: "0.28em",
          opacity: 0.85,
        }}
      >
        {card.modality}
      </p>

      <h4
        className="line-clamp-3 font-serif leading-tight"
        style={{ fontSize: "16px", color: "#2b241b" }}
      >
        {card.title}
      </h4>

      {(card.creator || card.year > 0) && (
        <>
          <div
            className="mx-auto my-3 h-px w-10"
            style={{ backgroundColor: "rgba(139, 111, 71, 0.35)" }}
          />
          <p
            className="font-serif italic"
            style={{
              fontSize: "11px",
              lineHeight: 1.5,
              opacity: 0.85,
            }}
          >
            {card.creator}
            {card.creator && card.year > 0 ? " · " : ""}
            {card.year > 0 ? card.year : ""}
          </p>
        </>
      )}
    </div>
  );
}

function PoemPoster({ excerpt }: { excerpt: string }) {
  const lines = (excerpt || "")
    .split("\n")
    .map((l) => l.trim())
    .filter((l) => l.length > 0)
    .slice(0, 6);

  return (
    <div
      className="flex h-full w-full flex-col justify-center overflow-hidden px-4 py-5 text-center font-serif italic"
      style={{
        backgroundColor: "#EFE8D4",
        color: "#3a332b",
        fontSize: "13px",
        lineHeight: 1.8,
        boxShadow: "inset 0 0 40px rgba(80, 60, 40, 0.08)",
      }}
    >
      {lines.map((line, i) => (
        <p
          key={i}
          className="m-0 py-0.5"
          style={{
            whiteSpace: "nowrap",
            overflow: "hidden",
            textOverflow: "ellipsis",
          }}
          title={line}
        >
          {line}
        </p>
      ))}
    </div>
  );
}

function PoemModal({
  card,
  onClose,
}: {
  card: ProductCard;
  onClose: () => void;
}) {
  const [lines, setLines] = useState<string[] | null>(null);
  const [fetchErr, setFetchErr] = useState<string | null>(null);

  useEffect(() => {
    let alive = true;
    setLines(null);
    setFetchErr(null);
    // PoetryDB supports CORS; fall back to the cached excerpt on any failure.
    fetch(card.external_url)
      .then((r) => {
        if (!r.ok) throw new Error(`poetrydb ${r.status}`);
        return r.json();
      })
      .then((data) => {
        if (!alive) return;
        if (
          Array.isArray(data) &&
          data.length > 0 &&
          Array.isArray(data[0].lines)
        ) {
          setLines(data[0].lines as string[]);
        } else {
          throw new Error("unexpected response shape");
        }
      })
      .catch((e) => {
        if (!alive) return;
        setFetchErr(String(e));
        setLines((card.excerpt ?? "").split("\n"));
      });
    return () => {
      alive = false;
    };
  }, [card.external_url, card.excerpt]);

  useEffect(() => {
    function onKey(e: KeyboardEvent) {
      if (e.key === "Escape") onClose();
    }
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [onClose]);

  return (
    <div
      className="fixed inset-0 z-50 flex items-center justify-center px-4 py-8"
      style={{ backgroundColor: "rgba(0, 0, 0, 0.4)" }}
      onClick={onClose}
    >
      <div
        className="max-h-[85vh] w-full max-w-xl overflow-y-auto rounded-md bg-white px-8 py-9 shadow-2xl"
        onClick={(e) => e.stopPropagation()}
      >
        <h2 className="font-serif text-2xl leading-tight text-ink">
          {card.title}
        </h2>
        <p className="mt-1 mb-5 text-sm text-ink/60">{card.creator}</p>

        {lines === null ? (
          <div className="py-10 text-center font-serif italic text-ink/40">
            loading…
          </div>
        ) : (
          <div
            className="border-y border-muted py-6 text-center font-serif italic"
            style={{ color: "#3a332b", fontSize: "16px", lineHeight: 1.95 }}
          >
            {lines.map((line, i) => (
              <p
                key={i}
                className={line.trim() === "" ? "h-3" : "m-0 py-0.5"}
              >
                {line}
              </p>
            ))}
          </div>
        )}

        {fetchErr && lines && (
          <p className="mt-3 text-center text-xs italic text-ink/40">
            couldn't fetch full text; showing stored excerpt instead.
          </p>
        )}

        <div className="mt-6 flex justify-end">
          <button
            onClick={onClose}
            className="rounded border border-muted px-4 py-1.5 text-sm text-ink/70 transition hover:border-accent hover:text-accent"
          >
            close
          </button>
        </div>
      </div>
    </div>
  );
}
