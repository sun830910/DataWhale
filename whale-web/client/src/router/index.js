import Vue from 'vue';
import VueRouter from 'vue-router';
import Layout from '@/layout';

Vue.use(VueRouter);

const routes = [
  {
    path: '/',
    name: 'Home',
    component: Layout,
    redirect: '/home',
    children: [
      {
        path: 'home',
        name: 'home',
        component: () => import('../views/Home.vue'),
      },
      {
        path: 'origin',
        name: 'origin',
        // articles/editor/add
        component: () => import('../views/Origin.vue'),
      },
      {
        path: 'articles',
        name: 'articles',
        redirect: '/articles/list',
        component: () => import('../views/Articles.vue'),
        children: [
          {
            path: 'list',
            component: () => import('../views/ArticleList.vue'),
          },
          {
            path: 'detail/:id',
            component: () => import('../views/ArticleDetail.vue'),
          },
          {
            path: 'editor/:id',
            component: () => import('../views/ArticleEditor.vue'),
          },
        ],
      },
      {
        path: 'chat',
        name: 'chat',
        component: () => import('../views/Chat.vue'),
      },
      {
        path: 'rank',
        name: 'rank',
        component: () => import('../views/Rank.vue'),
      },
      {
        path: 'login',
        name: 'login',
        // route level code-splitting
        // this generates a separate chunk (about.[hash].js) for this route
        // which is lazy-loaded when the route is visited.
        component: () => import(/* webpackChunkName: "login" */ '../views/Login.vue'),
      },
      {
        path: 'register',
        name: 'register',
        // route level code-splitting
        // this generates a separate chunk (about.[hash].js) for this route
        // which is lazy-loaded when the route is visited.
        component: () => import(/* webpackChunkName: "register" */ '../views/Register.vue'),
      },
    ],
  },
  {
    path: '/verify/:token',
    name: 'verifyToken',
    // route level code-splitting
    // this generates a separate chunk (about.[hash].js) for this route
    // which is lazy-loaded when the route is visited.
    component: () => import(/* webpackChunkName: "verify" */ '../views/VerifyToken.vue'),
  },
];

const router = new VueRouter({
  mode: 'history',
  base: process.env.BASE_URL,
  routes,
});

export default router;
