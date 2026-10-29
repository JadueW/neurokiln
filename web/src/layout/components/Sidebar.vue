<template>
  <aside class="sidebar" :style="{ width: collapsed ? '64px' : '220px' }">
    <div class="logo" @click="$router.push('/dashboard')">
      <svg viewBox="0 0 32 32" width="28" height="28" aria-hidden="true">
        <path d="M12 3h8v3h-2v6.2l7.4 12.1A3 3 0 0 1 22.8 29H9.2a3 3 0 0 1-2.6-4.7L14 12.2V6h-2V3z" fill="#409eff"/>
        <path d="M10.5 21h11l2.2 3.6a1.2 1.2 0 0 1-1 1.9H9.3a1.2 1.2 0 0 1-1-1.9L10.5 21z" fill="#a0cfff"/>
      </svg>
      <span v-show="!collapsed">NeuroKiln 脑炉</span>
    </div>
    <el-scrollbar class="menu-scroll">
      <el-menu
        :default-active="$route.path"
        :collapse="collapsed"
        :collapse-transition="false"
        background-color="#304156"
        text-color="#bfcbd9"
        active-text-color="#409eff"
        router
      >
        <el-menu-item v-for="item in menus" :key="item.path" :index="'/' + item.path">
          <el-icon><component :is="item.meta.icon" /></el-icon>
          <template #title>{{ $t(item.meta.titleKey) }}</template>
        </el-menu-item>
      </el-menu>
    </el-scrollbar>
    <div v-show="!collapsed" class="sidebar-footer">{{ $t('menu.footer') }}</div>
  </aside>
</template>

<script setup>
import { menuRoutes } from '@/router'

defineProps({ collapsed: Boolean })
const menus = menuRoutes
</script>

<style scoped>
.sidebar {
  position: fixed; left: 0; top: 0; bottom: 0; z-index: 1001;
  background: var(--sidebar-bg); display: flex; flex-direction: column;
  transition: width 0.2s ease;
}
.logo {
  height: 56px; display: flex; align-items: center; gap: 10px; padding: 0 16px;
  color: #fff; font-weight: 600; font-size: 15px; cursor: pointer; white-space: nowrap;
  background: #2b3a4f;
}
.menu-scroll { flex: 1; }
.el-menu { border-right: none; }
.el-menu-item:hover { background: #26384d !important; }
.el-menu-item.is-active { background: #26384d !important; }
.sidebar-footer { padding: 12px 16px; color: #7f8ea3; font-size: 12px; }
</style>
