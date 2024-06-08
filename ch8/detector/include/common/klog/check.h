#pragma once

//DEFINE_CHECK_OP_IMPL(Check_EQ, ==)  // Compilation error with CHECK_EQ(NULL, x)?
//DEFINE_CHECK_OP_IMPL(Check_NE, !=)  // Use CHECK(x == NULL) instead.
//DEFINE_CHECK_OP_IMPL(Check_LE, <=)
//DEFINE_CHECK_OP_IMPL(Check_LT, < )
//DEFINE_CHECK_OP_IMPL(Check_GE, >=)
//DEFINE_CHECK_OP_IMPL(Check_GT, > )

//__FILE__, __LINE__

#include <cstdlib>

#define CHECK(expr) LOG_CHECK(expr)

#define CHECK_OP(expr1, op, expr2) CHECK((expr1) op (expr2))

#define CHECK_EQ(expr1, expr2) CHECK_OP(expr1, ==, expr2)
#define CHECK_NE(expr1, expr2) CHECK_OP(expr1, !=, expr2)
#define CHECK_LE(expr1, expr2) CHECK_OP(expr1, <=, expr2)
#define CHECK_LT(expr1, expr2) CHECK_OP(expr1, <, expr2)
#define CHECK_GE(expr1, expr2) CHECK_OP(expr1, >=, expr2)
#define CHECK_GT(expr1, expr2) CHECK_OP(expr1, >, expr2)

#define CHECK_NOTNULL(expr) CHECK(expr); expr

#define DCHECK(expr) DLOG_CHECK(expr)

#define DCHECK_OP(expr1, op, expr2) DCHECK(expr1 op expr2)

#define DCHECK_EQ(expr1, expr2) DCHECK_OP(expr1, ==, expr2)
#define DCHECK_NE(expr1, expr2) DCHECK_OP(expr1, !=, expr2)
#define DCHECK_LE(expr1, expr2) DCHECK_OP(expr1, <=, expr2)
#define DCHECK_LT(expr1, expr2) DCHECK_OP(expr1, <, expr2)
#define DCHECK_GE(expr1, expr2) DCHECK_OP(expr1, >=, expr2)
#define DCHECK_GT(expr1, expr2) DCHECK_OP(expr1, >, expr2)

