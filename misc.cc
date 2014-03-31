// vim:set fdm=marker:
struct DisjointSet/*{{{*/
{
  vector<int> parent;

  int root(int x)
  {
    if (parent[x] < 0) {
      return x;
    } else {
      parent[x] = root(parent[x]);
      return parent[x];
    }
  }

  explicit DisjointSet(int n) : parent(n, -1) {}

  bool unite(int x, int y)
  {
    const int a = root(x);
    const int b = root(y);
    if (a != b) {
      if (parent[a] < parent[b]) {
        parent[a] += parent[b];
        parent[b] = a;
      } else {
        parent[b] += parent[a];
        parent[a] = b;
      }
      return true;
    } else {
      return false;
    }
  }

  bool find(int x, int y) { return root(x) == root(y); }
  int size(int x) { return -parent[root(x)]; }
};/*}}}*/

// 1-origin であることに注意．
// bit.add(1, 1); bit.add(2, 2);  bit.add(3, 3) とすると
// - tree[0] = 0
// - tree[1] = 1
// - tree[2] = 3
// - tree[3] = 6
// となる．
template <class T>
struct BinaryIndexedTree/*{{{*/
{
  vector<T> tree;
  const int size;
  BinaryIndexedTree(int s) : tree(t), size(s) {}
  // i 番目までの要素の累積和
  T read(int i) const
  {
    T sum = 0;
    while (i > 0) {
      sum += tree[i];
      i -= i & -i;
    }
    return sum;
  }

  // i 番目の要素
  T read_single(int i) const
  {
    T sum = tree[i];
    if (i > 0) {
      const int z = i - (i & -i);
      --i;
      while (i != z) {
        sum -= tree[i];
        i -= (i & -i);
      }
    }
    return sum;
  }

  void add(int i, T v)
  {
    while (i <= size) {
      tree[i] += v;
      i += (i & -i);
    }
  }

  // read(i) == vとなる最小のi。存在しなければ-1。
  int search(T v) {
        int left = 0, right = size;
        while(left+1 < right) {
            const int center = (left+right) / 2;
            if(read(center) < v) {
                left = center;
            } else {
                right = center;
            }
        }
        if(right == size || read(right) != v) return -1;
        return right;
    }
};/*}}}*/

// POJ 3264 Balanced Lineup
// AOJ 2431 House Moving
template <class T, class Compare>
struct SegmentTree/*{{{*/
{
  vector<T>& mem;
  vector<int> indexes;
  Compare cmp;
  SegmentTree(vector<T>& cs)
    : mem(cs), indexes(4*cs.size(), -1)
  {
    build(0, 0, cs.size());
  }

  void build(int idx, int left, int right)
  {
    if (left+1 == right) {
      indexes[idx] = left;
    } else {
      const int mid = (left + right)/2;
      build(2*idx+1, left, mid);
      build(2*idx+2, mid, right);
      // minimum in [left, right)
      if (cmp(mem[indexes[2*idx+1]], mem[indexes[2*idx+2]])) {
        indexes[idx] = indexes[2*idx+1];
      } else {
        indexes[idx] = indexes[2*idx+2];
      }
    }
  }

  inline T query_value(int left, int right) const { return mem[query_index(left, right)]; }

  inline int query_index(int left, int right) const { return query_index(left, right, 0, 0, mem.size()); }

  int query_index(int left, int right, int i, int a, int b) const
  {
    // [a, b) is the range of indexes[i]
    if (b <= left || right <= a) {
      // does not intersect
      return -1;
    } else if (left <= a && b <= right) {
      // contains
      return indexes[i];
    } else {
      const int m = (a+b)/2;
      const int l = query_index(left, right, 2*i+1, a, m);
      const int r = query_index(left, right, 2*i+2, m, b);
      if (l == -1) {
        return r;
      } else if (r == -1) {
        return l;
      } else {
        if (cmp(mem[l], mem[r])) {
          return l;
        } else {
          return r;
        }
      }
    }
  }

  void update(int idx, T val)
  {
    mem[idx] = val;
    update_index(0, mem.size(), 0, idx);
  }

  void update_index(int left, int right, int i, int idx)
  {
    if (left+1 == right) {
      //indexes[i] = idx;
    } else {
      const int mid = (left+right)/2;
      if (idx < mid) {
        update_index(left, mid, 2*i+1, idx);
      } else {
        update_index(mid, right, 2*i+2, idx);
      }
      if (cmp(mem[indexes[2*i+1]], mem[indexes[2*i+2]])) {
        indexes[i] = indexes[2*i+1];
      } else {
        indexes[i] = indexes[2*i+2];
      }
    }
  }

};/*}}}*/

// AOJ 2331 A Way to Inveite Friends
// POJ 2777 Count Color
// COCI 2012/2013 #6 Burek
// Codeforces #174(Div.1)A Cows and Sequence
// 更新も O(log N) で可能な SegmentTree．
template <class T>
struct SegmentTree/*{{{*/
{
    vector<T> nodes, stocks;
    int size;
    SegmentTree(int size) : size(size) {
        nodes.resize(size*4, 0);
        stocks.resize(size*4, 0);
    }

    void maintain_consistency(size_t pos) {
        if(stocks[pos] != 0) {
            // CAUTION: These expressions depend on following constraint:
            //  size = 2 ** N
            if(pos*2+1 < stocks.size()) stocks[pos*2+1] += stocks[pos] / 2;
            if(pos*2+2 < stocks.size()) stocks[pos*2+2] += stocks[pos] / 2;
            nodes[pos] += stocks[pos];
            stocks[pos] = 0;
        }
    }

    // [left, right) に対するクエリ．
    // 現在のノードはpos で， [pl, pr) を表わしている．
    T get_inner(int left, int right, size_t pos, int pl, int pr) {
        if(pr <= left || right <= pl) return 0; // 交差しない
        if(left <= pl && pr <= right) return nodes[pos] + stocks[pos]; // 完全に含まれる

        maintain_consistency(pos);

        const int center = (pl+pr) / 2;
        T lv = get_inner(left, right, pos*2+1, pl, center);
        T rv = get_inner(left, right, pos*2+2, center, pr);
        return lv + rv;
    }

    T get(int left, int right) {
        return get_inner(left, right, 0, 0, size);
    }

    T add_inner(int left, int right, size_t pos, int pl, int pr, T val) {
        if(pr <= left || right <= pl) { // 交差しない
            if(pos >= nodes.size()) return 0;
            else return stocks[pos] + nodes[pos];
        }
        if(left <= pl && pr <= right) {
            stocks[pos] += (pr-pl) * val;
            return stocks[pos] + nodes[pos]; // 完全に含まれる
        }

        maintain_consistency(pos);

        const int center = (pl+pr)/2;
        T lv = add_inner(left, right, pos*2+1, pl, center, val);
        T rv = add_inner(left, right, pos*2+2, center, pr, val);
        return nodes[pos] = lv+rv;
    }

    // Update range [left, right) in O(log N).
    T add(int left, int right, T val) {
        return add_inner(left, right, 0, 0, size, val);
    }
};/*}}}*/

// honeycomb {{{
/*
 * 0:  a a a a a a a a
 * 1:   b b b b b b b
 * 2:  a a a a a a a a
 * のように，左上(0,0)が出張っている場合の配列．
 * 左上が引っ込んでいる場合はEvenとOddを逆にする．
 */
const int DR[6] = {0, -1, -1, 0, 1, 1};
const int DC[2][6] = {
    {-1, -1, 0, 1, 0, -1}, // Even
    {-1, 0, 1, 1, 1, 0}, // Odd
};
// }}}

// Suffix Array {{{
struct SAComp {
    const vector<int> *grp;
    int h;
    SAComp(const vector<int> *grp, int h) : grp(grp), h(h) {}

    bool operator ()(int a, int b) const {
        int va = grp->at(a);
        int vb = grp->at(b);
        int vah = a+h < grp->size() ? grp->at(a+h) : INT_MIN;
        int vbh = b+h < grp->size() ? grp->at(b+h) : INT_MIN;
        return (va == vb) ? vah < vbh : va < vb;
    }
};

// Suffix Arrayを構築する．
// A Fast Algorithm for Making Suffix Arrays and for Burrows-Wheeler Transformation
// (Kunihiko Sadakane, 1998)
// の実装．ただし，以下の変更を加えている．
// ・同じグループごとにソートするのではなく，Suffix Array全体を一度にソートする．
// saの中身は開始インデックス．
//
// 計算量O(N (log N)^2)
void suffix_array(const string &str, vector<int> &sa) {
    assert(sa.size() >= str.size());

    int N = str.size();
    vector<int> group(N, 0), next(N, 0);
    for(int i = 0; i < N; ++i) {
        sa[i] = i;
        group[i] = str[i];
    }
    {
        SAComp cmp(&group, 0);
        sort(sa.begin(), sa.end(), cmp);
        next[sa[0]] = 0;
        for(int i = 1; i < N; ++i) {
            next[sa[i]] = next[sa[i-1]] + cmp(sa[i-1], sa[i]);
        }
        group.swap(next);
    }

    for(int h = 1; h < N && group[N-1] != N-1; h <<= 1) {
        //Generate <_{2*h} ordered array from <_{h} ordered array
        //この中身はcmpのコンストラクタ引数以外，上のブロックと同じ．
        SAComp cmp(&group, h);
        sort(sa.begin(), sa.end(), cmp);
        next[sa[0]] = 0;
        for(int i = 1; i < N; ++i) {
            next[sa[i]] = next[sa[i-1]] + cmp(sa[i-1], sa[i]);
        }
        group.swap(next);
    }
}

// Longest Common Prefixを計算する。
// lcpa[i] = sa[i]とsa[i+1]のLCP長。
// O(N)
// POJ3882 Stammering Aliens
void lcp(const string &str, const vector<int> &sa, vector<int> &lcpa) {
    const int N = str.size();
    vector<int> inv(N);
    for(int i = 0; i < N; ++i) {
        inv[sa[i]] = i;
    }
    int h = 0;
    for(int i = 0; i < N; ++i) {
        const int next = inv[i]+1 < N ? sa[inv[i]+1] : -1;
        if(next == -1) {
            h = 0;
            lcpa[inv[i]] = -1;
        } else {
            if(h > 0) --h;
            const int lim = min(N-i, N-next);
            for(int j = h; j < lim; ++j) {
                if(str[i+j] != str[next+j]) break;
                ++h;
            }
            lcpa[inv[i]] = h;
        }
    }
}

// 文字列を検索する．
// 複数の候補がある場合，最初に一致したインデックスを返す．
// 計算量O(M log N)
int find(const string &src, const string &str, const vector<int> &sa) {
    int left = 0, right = sa.size();
    while(left < right) {
        int mid = (left+right)/2;
        int res = src.compare(sa[mid], min(src.size()-sa[mid], str.size()), str);
        if(res == 0) return mid;
        if(res < 0) left = mid+1;
        else right = mid;
    }
    return -1;
}/*}}}*/
