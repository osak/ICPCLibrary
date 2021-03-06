// vim:set fdm=marker:
#include <vector>
#include <queue>
#include <limits>
using namespace std;

struct edge {
  int u, v;
  typedef int weight_type;
  weight_type w;
  edge(int s, int d, weight_type w_) : u(s), v(d), w(w_) {}
};

// POJ 3259 Wormholes
template <typename Edge>
pair<vector<typename Edge::weight_type>, bool>
bellman_ford(const vector<Edge>& g,/*{{{*/
    typename vector<Edge>::size_type N,
    typename vector<Edge>::size_type start,
    typename Edge::weight_type inf = 1000000)
{
  /* Edge must have
   * - u (source node)
   * - v (dest node)
   * - w (weight)
   * - weight_type
   */
  typedef typename vector<Edge>::size_type size_type;
  typedef typename Edge::weight_type weight_type;
  vector<weight_type> dist(N, inf);
  dist[start] = 0;

  for (size_type i = 0; i < N; i++) {
    for (typename vector<Edge>::const_iterator it(g.begin()); it != g.end(); ++it) {
      if (dist[it->u] + it->w < dist[it->v]) {
        dist[it->v] = dist[it->u] + it->w;
      }
    }
  }

  for (typename vector<Edge>::const_iterator it(g.begin()); it != g.end(); ++it) {
    if (dist[it->u] + it->w < dist[it->v]) {
      return make_pair(dist, false);
    }
  }
  return make_pair(dist, true);
}/*}}}*/

// O(E^2 V)
template <typename T>
T edmonds_karp(const vector<vector<T> >& capacity, int source, int sink)/*{{{*/
{
  const int N = capacity.size();
  vector<vector<T> > flow(N, vector<T>(N, 0));
  T max_flow = 0;

  while (true) {
    vector<int> parent(N, -1);
    queue<int> q;
    q.push(source);

    while (!q.empty() && parent[sink] < 0) {
      const int v = q.front();
      q.pop();

      for (int u = 0; u < N; u++) {
        if (parent[u] < 0 && capacity[v][u] - flow[v][u] > 0) {
          parent[u] = v;
          if (u == sink) {
            break;
          }
          q.push(u);
        }
      }
    }

    if (parent[sink] < 0) {
      break;
    }

    T aug = numeric_limits<T>::max();
    for (int v = sink; v != source; v = parent[v]) {
      const int u = parent[v];
      aug = min(aug, capacity[u][v] - flow[u][v]);
    }
    max_flow += aug;
    for (int v = sink; v != source; v = parent[v]) {
      const int u = parent[v];
      flow[u][v] += aug;
      flow[v][u] -= aug;
    }
  }

  return max_flow;
}/*}}}*/

template <class T>
T dinic_augment(vector<vector<Edge<T>*>> &graph, vector<int> &level, vector<bool> &finished, int u, int sink, T cur)/*{{{*/
{
    if (u == sink || cur == 0) {
        return cur;
    }
    if (finished[u]) {
        return 0;
    }
    finished[u] = true;
    for(auto e : graph[u]) {
        if(e->capacity - e->flow > 0 && level[e->to] > level[u]) {
            const T f = dinic_augment(graph, level, finished, e->to, sink, min(cur, e->capacity - e->flow));
            if (f > 0) {
                e->flow += f;
                e->back->flow -= f;
                finished[u] = false;
                return f;
            }
        }
    }
    return 0;
}/*}}}*/

// O(V^2 E)
template <typename T>
T dinic(vector<vector<Edge<T>*>> &graph, int source, int sink)/*{{{*/
{
    const int N = graph.size();
    T max_flow = 0;

    vector<int> level(N);
    vector<bool> finished(N);
    while (true) {
        fill(level.begin(), level.end(), -1);
        level[source] = 0;
        queue<int> q;
        q.push(source);

        int d = N;
        while (!q.empty() && level[q.front()] < d) {
            const int u = q.front();
            q.pop();

            if (u == sink) {
                d = level[u];
            }
            for(auto e : graph[u]) {
                if (level[e->to] < 0 && e->capacity - e->flow > 0) {
                    q.push(e->to);
                    level[e->to] = level[u] + 1;
                }
            }
        }

        fill(finished.begin(), finished.end(), false);
        bool updated = false;
        while (true) {
            const T f = dinic_augment<T>(graph, level, finished, source, sink, INF);
            if (f == 0) {
                break;
            }
            max_flow += f;
            updated = true;
        }

        if (!updated) {
            break;
        }
    }

    return max_flow;
}/*}}}*/

// 連結な無向グラフのすべての辺を通るような閉路で，コストの合計の最小値
// O(N^2 2^N)
template <class T>
T chinese_postman(const vector<vector<pair<int,T> > >& g)/*{{{*/
{
  T total = 0;
  vector<int> odd_nodes;
  for (int i = 0; i < static_cast<int>(g.size()); i++) {
    for (vector<pair<int,T> >::const_iterator it(g[i].begin()); it != g[i].end(); ++it) {
      total += it->second;
    }
    if (g[i].size() % 2 == 1) {
      odd_nodes.push_back(i);
    }
  }
  total /= 2;

  const int N = odd_nodes.size();
  vector<vector<T> > w(N, vector<T>(N, -1));
  for (int i = 0; i < N; i++) {
    // dijkstra
    vector<T> dist(g.size(), 1000000);
    dist[odd_nodes[i]] = 0;
    priority_queue<pair<T,int> > q;
    q.push(make_pair(0, odd_nodes[i]));
    while (!q.empty()) {
      const T cost = -q.top().first;
      const int n = q.top().second;
      q.pop();
      for (vector<pair<int,T> >::const_iterator it(g[n].begin()); it != g[n].end(); ++it) {
        const T c = cost + it->second;
        if (c < dist[it->first]) {
          dist[it->first] = c;
          q.push(make_pair(-c, it->first));
        }
      }
    }
    for (int j = 0; j < N; j++) {
      w[i][j] = dist[odd_nodes[j]];
    }
  }

  vector<T> dp(1<<N, 1000000);
  dp[0] = 0;
  for (int s = 0; s < (1<<N); s++) {
    for (int i = 0; i < N; i++) {
      if (s & (1<<i)) {
        continue;
      }
      for (int j = i+1; j < N; j++) {
        if (s & (1<<j)) {
          continue;
        }
        dp[s | (1<<i) | (1<<j)] = min(dp[s | (1<<i) | (1<<j)], dp[s] + w[i][j]);
      }
    }
  }
  return total + dp[(1<<N)-1];
}/*}}}*/

bool bm_augment(const vector<vector<int> >& g, int u, vector<int>& match_to, vector<bool>& visited) // {{{
{
  if (u < 0) {
    return true;
  }

  for (vector<int>::const_iterator it(g[u].begin()); it != g[u].end(); ++it) {
    if (!visited[*it]) {
      visited[*it] = true;
      if (bm_augment(g, match_to[*it], match_to, visited)) {
        match_to[u] = *it;
        match_to[*it] = u;
        return true;
      }
    }
  }
  return false;
} // }}}

// O(V(V+E))
// Ford Fulkerson の変形．
int bipartite_matching(const vector<vector<int> >& g, int L, vector<pair<int,int> >& matching)  // {{{
{
  const int N = g.size();
  vector<int> match_to(N, -1);
  int match = 0;
  for (int u = 0; u < N; u++) {
    vector<bool> visited(N, false);
    if (bm_augment(g, u, match_to, visited)) {
      match++;
    }
  }
  for (int u = 0; u < L; u++) {
    if (match_to[u] >= 0) {
      matching.push_back(make_pair(u, match_to[u]));
    }
  }
  return match;
} // }}}

// mininum cut
// O(V^3)
template <class T>
T stoer_wagner(vector<vector<T> > g)/*{{{*/
{
  const int N = g.size();
  vector<int> v(N);
  for (int i = 0; i < N; i++) {
    v[i] = i;
  }

  T cut = numeric_limits<T>::max();
  for (int m = N; m > 1; m--) {
    vector<T> ws(m, 0);
    int s, t = 0;
    T w;
    for (int k = 0; k < m; k++) {
      s = t;
      t = distance(ws.begin(), max_element(ws.begin(), ws.end()));
      w = ws[t];
      ws[t] = -1;
      for (int i = 0; i < m; i++) {
        if (ws[i] >= 0) {
          ws[i] += g[v[t]][v[i]];
        }
      }
    }
    for (int i = 0; i < m; i++) {
      g[v[i]][v[s]] += g[v[i]][v[t]];
      g[v[s]][v[i]] += g[v[t]][v[i]];
    }
    v.erase(v.begin() + t);
    cut = min(cut, w);
  }
  return cut;
}/*}}}*/

// O(V+E)
// 強連結成分分解 {{{
void dfs1(int pos, const vector<vector<int>> &g, vector<bool> &visited, vector<int> &buf) {
    if(visited[pos]) return;
    visited[pos] = true;
    for(int to : g[pos]) dfs1(to, g, visited, buf);
    buf.push_back(pos);
}

void dfs2(int pos, const vector<vector<int>> &g, int label, vector<int> &buf) {
    if(buf[pos] != -1) return;
    buf[pos] = label;
    for(int to : g[pos]) dfs2(to, g, label, buf);
}

vector<int> scc(const vector<vector<int>> &g) {
    const int N = g.size();
    vector<vector<int>> rev(N);
    for(int i = 0; i < N; ++i) {
        for(int to : g[i]) {
            rev[to].push_back(i);
        }
    }
    vector<bool> visited(N, false);
    vector<int> ord;
    for(int i = 0; i < N; ++i) {
        dfs1(i, g, visited, ord);
    }
    reverse(begin(ord), end(ord));
    vector<int> label(N, -1);
    int l = 0;
    for(int pos : ord) {
        if(label[pos] == -1) dfs2(pos, rev, l++, label);
    }
    return label;
} // }}}

// Tarjan の橋分解アルゴリズム {{{
void dfs(const vector<vector<Edge*>> &graph, int pos, int prev, vector<int> &low, vector<int> &ord, vector<int> &buf, int &cnt) {
    ord[pos] = cnt++;
    low[pos] = ord[pos];
    if(start_of[pos] == -1) return;
    for(Edge *e : graph[pos]) {
        const int next = e->to;
        if(ord[next] == -1) {
            dfs(graph, next, pos, low, ord, buf, cnt);
            low[pos] = min(low[pos], low[next]);
            if(low[next] == ord[next]) {
                buf.push_back(i);
            }
        } else if(next != prev) {
            low[pos] = min(low[pos], ord[next]);
        }
    }
}

/*
 * Tarjan の橋分解アルゴリズム。
 * O(V+E)
 *
 * POJ3177 Redundant Paths
 */
void bridges(const vector<vector<Edge*>> &graph, vector<int> &buf) {
    const int N = graph.size();
    vector<int> low(N, -1);
    vector<int> ord(N, -1);
    int cnt = 0;
    TIMES(i, N) {
        if(ord[i] == -1) {
            dfs(graph, i, -1, low, ord, buf, cnt);
        }
    }
}// }}}

/* 2-SAT
 * i 番目の正のリテラルは i<<1 、負のリテラルは (i<<1)|1 で表現する。
 * リテラルの否定は 1 と XOR をとるだけ。
 *
 * (A /\ B) == (!A -> A) /\ (!B -> B)
 * (A \/ B) == (!A -> B) /\ (!B -> A)
 *
 * POJ 3678 Katu Puzzle
 * AOJ 2504 Statement Coverage
 */
bool two_sat(const vector<vector<int> >& g)/*{{{*/
{
  const int N = g.size()/2;
  const pair<vector<int>,int> p = strongly_connected_components(g);
  const vector<int>& scc_map = p.first;
  for (int i = 0; i < N; i++) {
    if (scc_map[i<<1] == scc_map[(i<<1)|1]) {
      return false;
    }
  }
  return true;
}/*}}}*/

// minimum cost flow
// returns (cost, flow)
// POJ 2195 Going Home
struct Edge/*{{{*/
{
  int index;
  int capacity;
  int cost;
  Edge(int i, int c, int d) : index(i), capacity(c), cost(d) {}
};/*}}}*/

// O(V^2 U C) where
//  U = sum of capacity
//  C = sum of cost
pair<int,int> primal_dual(const vector<vector<Edge> >& g, int source, int sink)/*{{{*/
{
  const int N = g.size();
  vector<vector<int> > capacity(N, vector<int>(N, 0)), cost(N, vector<int>(N, 0)), flow(N, vector<int>(N, 0));
  for (int i = 0; i < N; i++) {
    for (vector<Edge>::const_iterator it(g[i].begin()); it != g[i].end(); ++it) {
      capacity[i][it->index] += it->capacity;
      cost[i][it->index] += it->cost;
    }
  }
  pair<int,int> total;  // (cost, flow)
  vector<int> h(N, 0);
  for (int f = numeric_limits<int>::max(); f > 0; ) {
    vector<int> dist(N, 1000000);
    dist[source] = 0;
    vector<int> parent(N, -1);
    priority_queue<pair<int,int> > q;
    q.push(make_pair(0, source));
    while (!q.empty()) {
      const int n = q.top().second;
      const int c = -q.top().first;
      q.pop();
      for (vector<Edge>::const_iterator it(g[n].begin()); it != g[n].end(); ++it) {
        if (capacity[n][it->index] - flow[n][it->index] > 0) {
          const int c2 = c + cost[n][it->index] + h[n] - h[it->index];
          if (c2 < dist[it->index]) {
            dist[it->index] = c2;
            parent[it->index] = n;
            q.push(make_pair(-c2, it->index));
          }
        }
      }
    }
    if (parent[sink] == -1) {
      break;
    }

    int e = f;
    for (int i = sink; i != source; i = parent[i]) {
      e = min(e, capacity[parent[i]][i] - flow[parent[i]][i]);
    }
    for (int i = sink; i != source; i = parent[i]) {
      total.first += e * cost[parent[i]][i];
      flow[parent[i]][i] += e;
      flow[i][parent[i]] -= e;
    }
    f -= e;
    total.second += e;
    for (int i = 0; i < N; i++) {
      h[i] += dist[i];
    }
  }
  return total;
}/*}}}*/

// 多重辺があっても動く Primal-Dual {{{
// POJ 2047 Concert Hall Scheduling
// Codeforces #170(Div.1)E Binary Tree on Plane
// Cost を浮動小数点数にするときは，EPS を考慮しないと Dijkstra 部で死ぬことがある．
template<class Flow, class Cost>
struct Edge {
    int from, to;
    Flow capacity, flow;
    Cost cost;
    Edge *back;
    Edge() {}
    Edge(int from, int to, Flow c, Cost d, Edge *b) : from(from), to(to), capacity(c), flow(0), cost(d), back(b) {}
};
template<class Flow, class Cost>
void make_edge(vector<vector<Edge<Flow,Cost>*>> &g, int src, int dst, Flow c, Cost d) {
    auto *e = new Edge<Flow,Cost>(src, dst, c, d, nullptr);
    auto *back = e->back = new Edge<Flow,Cost>(dst, src, 0, -d, e);
    g[src].push_back(e);
    g[dst].push_back(back);
}

template<class Flow, class Cost>
pair<Flow, Cost> primal_dual(vector<vector<Edge<Flow,Cost>*>> &g, int src, int sink, int max_flow) {
    const int N = g.size();
    pair<Flow, Cost> res;
    vector<Cost> h(N), dist(N);
    vector<Edge<Flow,Cost>*> parent(N);
    for(Flow f = max_flow; f > 0; ) {
        fill(dist.begin(), dist.end(), INF);
        dist[src] = 0;
        fill(parent.begin(), parent.end(), nullptr);
        priority_queue<pair<Cost,int>> q;
        q.push(make_pair(0, src));
        while(!q.empty()) {
            const int n = q.top().second;
            const Cost c = -q.top().first;
            q.pop();
            if(dist[n] < c) {
                continue;
            }
            for(auto e : g[n]) {
                if(e->capacity - e->flow > 0) {
                    const Cost c2 = c + e->cost + h[n] - h[e->to];
                    if(c2 < dist[e->to]) {
                        dist[e->to] = c2;
                        parent[e->to] = e;
                        q.push(make_pair(-c2, e->to));
                    }
                }
            }
        }
        if(parent[sink] == nullptr) {
            break;
        }

        Flow to_push = f;
        for(int i = sink; i != src; i = parent[i]->from) {
            auto e = parent[i];
            to_push = min(to_push, e->capacity - e->flow);
        }
        for(int i = sink; i != src; i = parent[i]->from) {
            auto e = parent[i];
            res.second += to_push * e->cost;
            e->flow += to_push;
            e->back->flow -= to_push;
        }
        f -= to_push;
        res.first += to_push;
        for(int i = 0; i < N; ++i) {
            h[i] += dist[i];
        }
    }
    return res;
}/*}}}*/
