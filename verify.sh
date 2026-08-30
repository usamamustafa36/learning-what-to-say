#!/usr/bin/env bash
# ONE GATE. Regenerates every figure and number from the stored results, rebuilds the manuscript,
# and reports the four things that must hold before submission:
#
#   10 pages   0 overfull boxes   0 undefined references   0 unresolved [NOT RUN] macros
#
# It does not run any experiment. If a sweep is unfinished, make_numbers.py withholds its macros and
# they surface here as [NOT RUN], which is the intended loud failure rather than a plausible value.
set -u
cd "$(dirname "$0")/.."
fail=0
say() { printf '%-42s %s\n' "$1" "$2"; }

( cd figures && PYTHONPATH=../code python3 make_figures.py ) || fail=1
( cd paper && PYTHONPATH=../code python3 make_numbers.py ) || fail=1

# numbers.tex must be a pure function of the results: regenerating twice must not change a byte.
cp paper/numbers.tex /tmp/numbers.first.$$
( cd paper && PYTHONPATH=../code python3 make_numbers.py >/dev/null ) || fail=1
if cmp -s /tmp/numbers.first.$$ paper/numbers.tex; then say "numbers.tex deterministic" "yes"
else say "numbers.tex deterministic" "NO -- it depends on something outside results/"; fail=1; fi
rm -f /tmp/numbers.first.$$

( cd paper && pdflatex -interaction=nonstopmode main.tex >/dev/null 2>&1 \
           && pdflatex -interaction=nonstopmode main.tex >/tmp/verify.$$.log 2>&1 )
log=/tmp/verify.$$.log

pages=$(pdfinfo paper/main.pdf | awk '/^Pages/{print $2}')
over=$(grep -c 'Overfull' "$log")
undef=$(grep -c 'undefined' "$log")
errs=$(grep -c '^!' "$log")
notrun=$(pdftotext paper/main.pdf - | grep -o 'NOT RUN' | wc -l)

say "pages"                  "$pages   (must be 10: IEEE TCCN charges \$220/page above ten)"
say "overfull boxes"         "$over"
say "undefined references"   "$undef"
say "LaTeX errors"           "$errs"
say "unresolved [NOT RUN]"   "$notrun"
[ "$pages" = 10 ] || fail=1
[ "$over" = 0 ] && [ "$undef" = 0 ] && [ "$errs" = 0 ] && [ "$notrun" = 0 ] || fail=1
rm -f "$log"

( cd paper && python3 audit_literals.py --check ) || fail=1

echo
if [ "$fail" = 0 ]; then echo "VERIFY: all gates pass. Run 'python3 qa.py' for the full suite."
else echo "VERIFY: FAILED -- see above."; fi
exit "$fail"
